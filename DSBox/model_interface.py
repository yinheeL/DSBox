import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


class BaseModel:
    """
Abstract base class: Any model used for selection must implement infer_one / infer_batch.
If you want to do LearningLoss, you also need to implement predict_loss_one / predict_loss_batch.
    """
    def infer_one(self, sample):
        raise NotImplementedError("Subclasses must implement infer_one")

    def infer_batch(self, samples, batch_size=16):
        raise NotImplementedError("Subclasses must implement infer_batch")

    def predict_loss_one(self, sample):
        raise NotImplementedError("Subclasses must implement predict_loss_one")

    def predict_loss_batch(self, samples, batch_size=16):
        raise NotImplementedError("Subclasses must implement predict_loss_batch")


class DevignCodeBERTModel(BaseModel, nn.Module):
    """
CodeBERT-based binary classification model + loss prediction head (LearningLoss), compatible with Trainer.
    """
    def __init__(self, device='cuda', checkpoint_path=None):
        nn.Module.__init__(self)
        self.device = torch.device(device)
        # Tokenizer and Main Model
        self.tokenizer = RobertaTokenizer.from_pretrained(
            checkpoint_path or "microsoft/codebert-base"
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            checkpoint_path or "microsoft/codebert-base",
            num_labels=2,
            use_safetensors=True
        ).to(self.device)

        # Encoder for feature extraction (infer_one/infer_batch)
        self.encoder = RobertaModel.from_pretrained(
            checkpoint_path or "microsoft/codebert-base",
            use_safetensors=True
        ).to(self.device)

        hidden_size = self.encoder.config.hidden_size
        # Classification head (reuse model.classifier)
        self.classifier = nn.Linear(hidden_size, 2).to(self.device)
        # Loss prediction head
        self.loss_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(self.device)

        # If there is a checkpoint_path, load the saved state_dict
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.encoder.load_state_dict(ckpt.get("encoder", {}))
            self.classifier.load_state_dict(ckpt.get("classifier", {}))
            self.loss_predictor.load_state_dict(ckpt.get("loss_predictor", {}))
            self.encoder.eval()
            self.classifier.eval()
            self.loss_predictor.eval()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
       Supports the forward method called by Trainer, calling the underlying RobertaForSequenceClassification.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def to(self, device):
        """Allow Trainer to call model.to(device) method to migrate all submodules to the device"""
        self.device = torch.device(device)
        super().to(self.device)
        return self

    def _encode(self, code_str):
        tokens = self.tokenizer(
            code_str,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]
        return cls_feat.squeeze(0)

    def infer_one(self, code_str):
        self.encoder.eval()
        self.classifier.eval()
        features = self._encode(code_str)
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        return probs

    def infer_batch(self, code_list, batch_size=16):
        from tqdm import tqdm
        all_probs = []
        for i in tqdm(range(0, len(code_list), batch_size), desc="Infer batch"):
            sub_list = code_list[i: i + batch_size]
            tokens = self.tokenizer(
                sub_list,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_feats = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_feats)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            all_probs.extend(probs)
        return all_probs

    def predict_loss_one(self, code_str):
        self.encoder.eval()
        self.loss_predictor.eval()
        features = self._encode(code_str)
        loss_pred = self.loss_predictor(features).squeeze(0)
        return loss_pred.item()

    def predict_loss_batch(self, code_list, batch_size=16):
        from tqdm import tqdm
        all_loss_preds = []
        for i in tqdm(range(0, len(code_list), batch_size), desc="Predict loss"):
            sub_list = code_list[i: i + batch_size]
            tokens = self.tokenizer(
                sub_list,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_feats = outputs.last_hidden_state[:, 0, :]
            loss_preds = self.loss_predictor(cls_feats).squeeze(1)
            all_loss_preds.extend(loss_preds.cpu().tolist())
        return all_loss_preds

# Other LoRA functions can be added as needed without affecting the Trainer interface


    def lora_warmup(self, dataset, epochs=3, batch_size=16, lr=2e-5):
        """
  Use PEFT LoRA to warmup the labeled data and save it in output/lora-warmup.
 Return to the directory path.
        """
        # 1) wrap with PEFT
        peft_conf = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8, lora_alpha=32, lora_dropout=0.1
        )
        lora_model = get_peft_model(self.model, peft_conf)
        lora_model.to(self.device).train()

        # 2) prepare Dataset/DataLoader
        class _DS(Dataset):
            def __init__(self, examples, tokenizer):
                self.examples = examples
                self.tokenizer = tokenizer

            def __len__(self): return len(self.examples)

            def __getitem__(self, idx):
                code, label = self.examples[idx]
                toks = self.tokenizer(
                    code, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
                )
                return {
                    "input_ids": toks.input_ids.squeeze(0),
                    "attention_mask": toks.attention_mask.squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long)
                }

        examples = dataset.get_labeled()
        ds = _DS(examples, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=False)

        optimizer = AdamW(lora_model.parameters(), lr=lr)

        # 3) manual training loop
        out_dir = os.path.abspath("output/lora-warmup")
        os.makedirs(out_dir, exist_ok=True)
        for ep in range(1, epochs + 1):
            total = 0.0
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labs = batch["labels"].to(self.device)

                outputs = lora_model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()

            print(f"[LoRA Warmup] Ep {ep}/{epochs} loss={total / len(loader):.4f}")

        lora_model.save_pretrained(out_dir)

        # replace base model
        self.model = lora_model
        return out_dir

    def lora_finetune(self, dataset, indices, epochs=2, batch_size=8, lr=1e-5):
        """
    Use LoRA to do a second fine-tune on the selected sample and save it in output/lora-finetune.
        """
        self.model.train()

        class _DS(Dataset):
            def __init__(self, examples, tokenizer):
                self.examples = examples
                self.tokenizer = tokenizer

            def __len__(self): return len(self.examples)

            def __getitem__(self, idx):
                code, label = self.examples[idx]
                toks = self.tokenizer(
                    code, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
                )
                return {
                    "input_ids": toks.input_ids.squeeze(0),
                    "attention_mask": toks.attention_mask.squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long)
                }

        examples = [(dataset.functions[i], dataset.full_labels[i]) for i in indices]
        ds = _DS(examples, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=False)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        out_dir = os.path.abspath("output/lora-finetune")
        os.makedirs(out_dir, exist_ok=True)
        for ep in range(1, epochs + 1):
            total = 0.0
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labs = batch["labels"].to(self.device)

                outputs = self.model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()

            print(f"[LoRA Finetune] Ep {ep}/{epochs} loss={total / len(loader):.4f}")

        self.model.save_pretrained(out_dir)

    def load_lora_adapter(self, adapter_dir: str):
        """
     Inject the LoRA adapter in the specified directory into the base model and return self for chain calls.
        """
        from peft import PeftConfig, PeftModel, get_peft_model


        peft_conf = PeftConfig.from_pretrained(adapter_dir)

        peft_model = get_peft_model(self.model, peft_conf)

        peft_model = PeftModel.from_pretrained(peft_model, adapter_dir)

        self.model = peft_model
        return self



    def get_effort_scores(self, batch_size: int = 32):
        """Return an effort/uncertainty score for every unlabeled sample.
        If `effort_head` exists, use it; otherwise fall back to classifier entropy margin.
        """
        abs_ids = list(self.dataset.unlabeled_indices)
        code_list = [self.dataset.functions[i] for i in abs_ids]

        scores: list[float] = []
        for i in range(0, len(code_list), batch_size):
            batch_codes = code_list[i : i + batch_size]
            toks = self.tokenizer(
                batch_codes,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                feats = self.encoder(**toks).last_hidden_state[:, 0, :]
                if hasattr(self, "effort_head"):
                    eff = self.effort_head(feats).squeeze(-1)  # shape=(bs,)
                else:
                    # fallback: use classifier uncertainty (1 - max prob)
                    logits = self.classifier(feats)
                    if logits.dim() == 2:
                        probs = torch.softmax(logits, dim=-1)
                        eff = 1.0 - torch.max(probs, dim=-1).values  # higher = harder
                    else:  # single‑logit model
                        eff = torch.sigmoid(logits).squeeze(-1)
                scores.extend(eff.cpu().tolist())
        return np.array(scores, dtype=float)

