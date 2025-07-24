import os
from typing import List
from sampling import GraphDensitySampler

import torch
import pickle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
    LlamaTokenizer,
)
from peft import PeftConfig
from datasets import load_dataset, load_from_disk
import itertools
import time
import math
import random
import json
import argparse
from tqdm import trange
import numpy as np

from effort_baseline_utils import (
    Effort_Trainer,
    save_feature,
    load_feature,
    save_time_cost,
)

# ------------ 这里是关键改动 ------------
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel,
)
# 新版 PEFT 中替代 prepare_model_for_int8_training
from peft.utils import prepare_model_for_kbit_training
# --------------------------------------

def get_moderate_index(data_score, rate=0.1):
    low = 0.5 - rate / 2
    high = 0.5 + rate / 2
    sorted_idx = data_score.argsort()
    low_idx = round(data_score.shape[0] * low)
    high_idx = round(data_score.shape[0] * high)
    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
    return ids

def get_ccs_index(data_score, rate=0.1):
    stratas = 50
    score = data_score
    total_num = int(len(data_score) * rate)
    min_score = torch.min(score)
    max_score = torch.max(score) * 1.0001
    step = (max_score - min_score) / stratas

    def bin_range(k):
        return min_score + k * step, min_score + (k + 1) * step


    strata_num = []
    for i in range(stratas):
        start, end = bin_range(i)
        num = torch.logical_and(score >= start, score < end).sum()
        strata_num.append(num)
    strata_num = torch.tensor(strata_num)

    def bin_allocate(num, bins):
        sorted_index = torch.argsort(bins)
        sort_bins = bins[sorted_index]
        num_bin = bins.shape[0]
        rest_exp_num = num
        budgets = []
        for i in range(num_bin):
            rest_bins = num_bin - i
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            budgets.append(cur_num)
            rest_exp_num -= cur_num
        rst = torch.zeros((num_bin,), dtype=torch.int)
        rst[sorted_index] = torch.tensor(budgets, dtype=torch.int)
        return rst

    budgets = bin_allocate(total_num, strata_num)


    selected_index = []
    sample_index = torch.arange(data_score.shape[0])
    for i in range(stratas):
        start, end = bin_range(i)
        mask = torch.logical_and(score >= start, score < end)
        pool = sample_index[mask]
        rand_index = torch.randperm(pool.shape[0])
        selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]
    return selected_index

def get_feature(base_model, dataset):
    feature_extractor = pipeline("feature-extraction", framework="pt", model=base_model)

    if not os.path.exists(dataset):
        data = load_dataset(dataset)
    else:
        data = load_from_disk(dataset)
    train_data = data["train"]
    raw_feature_list = []
    raw_target_list = []
    for item in trange(len(train_data)):
        mask = [label != -100 for label in train_data[item]["labels"]]
        target = [label for label, m in zip(train_data[item]["labels"], mask) if m]
        feat = feature_extractor(
            feature_extractor.tokenizer.decode(train_data[item]["input_ids"]),
            return_tensors="pt",
        )[0]
        mask.append(False)
        feat = feat[mask]
        raw_feature_list.append(feat.cpu())
        raw_target_list.append(target)
    return raw_feature_list, raw_target_list

def get_Moderate_score(raw_feature_list, raw_target_list):
    feature_list = torch.cat(raw_feature_list, dim=0).detach().numpy()
    target_list = np.array(list(itertools.chain(*raw_target_list)))
    classes_list = np.unique(target_list, axis=0)
    num_classes = len(classes_list)
    prot = np.zeros((num_classes, feature_list.shape[-1]))
    for i in trange(num_classes):
        prot[i] = np.median(
            feature_list[(target_list == classes_list[i]).nonzero(), :].squeeze(),
            axis=0,
        )
    prots_for_each_example = np.zeros((len(raw_feature_list), prot.shape[-1]))
    for i in trange(len(raw_feature_list)):
        idxs = [np.where(classes_list == tok)[0][0] for tok in raw_target_list[i]]
        prots_for_each_example[i, :] = np.sum(prot[idxs, :], axis=0)
    score_list = np.linalg.norm(raw_feature_list - prots_for_each_example, axis=1)
    return score_list

def get_D2_index(raw_features, data_score, rate=0.1, mis_ratio=0.4):
    total_num = len(data_score)
    coreset_num = int(rate * total_num)
    mis_num = int(mis_ratio * total_num)
    # 先剔除 “最难” 的 mis_num，再从剩余中做 coreset
    sorted_idx = data_score.argsort(descending=True)
    easy_idx = sorted_idx[mis_num:]
    features = np.array([f.mean(dim=0).numpy() for f in raw_features])
    sampler = GraphDensitySampler(
        X=features,
        y=None,
        gamma=0.1,
        seed=42,
        importance_scores=data_score,
    )
    coreset_index = sampler.select_batch_(coreset_num)
    return coreset_index

def get_baseline_score(
    base_model: str = "YOUR_DIR/test_code/wmt-gemma-7b",
    dataset: str = "wmt",
    tokenizer_path: str = "YOUR_DIR/models/gemma-7b",
    finetune: bool = False,
    method: List[str] = ["Effort"],
):
    train_data_path = dataset
    cutoff_len = 200
    lora_r = 81
    lora_alpha = 16
    lora_dropout = 0
    lora_target_modules = ["q_proj", "v_proj"]
    group_by_length = False
    gradient_accumulation_steps = 1
    device_map = "auto"
    os.environ["WANDB_DISABLED"] = "true"

    if finetune:
        config = PeftConfig.from_pretrained(base_model)
        if "QA" in base_model:
            raw_model = AutoModelForQuestionAnswering.from_pretrained(
                config.base_model_name_or_path, device_map="auto"
            )
        else:
            raw_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, device_map="auto"
            )
        model = PeftModel.from_pretrained(raw_model, base_model)

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, padding_side="left"
        )
    else:
        if "QA" in base_model:
            model = AutoModelForQuestionAnswering.from_pretrained(
                base_model, load_in_8bit=True, device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

    # ---- LoRA Config & Adapter ----
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="QUESTION_ANS" if "QA" in base_model else "CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, config)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


    if not os.path.exists(train_data_path):
        data = load_dataset(train_data_path)
    else:
        data = load_from_disk(train_data_path)
    train_data = data.train_test_split(test_size=0.1, seed=42)["train"]

    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Effort_Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=train_data,
        args=__import__("transformers").TrainingArguments(
            output_dir="./effort_output",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=1,
            learning_rate=0,
            fp16=False,
            logging_strategy="no",
            optim="adamw_torch",
            save_strategy="no",
            ddp_find_unused_parameters=None,
            group_by_length=group_by_length,
            report_to=None,
            gradient_checkpointing=False,
        ),
        data_collator=__import__("transformers").DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # 冻结除 LoRA 之外的所有层
    # trainer.freeze_layers(...)  # 按需启用

    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # ---- 计算并返回分数 ----
    score_dict = trainer.get_grad(method=method)
    return score_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--base_model", default="YOUR_DIR/wmt_gemma2", type=str)
    parser.add_argument("--tokenizer", default="YOUR_DIR/models/gemma-7b", type=str)
    parser.add_argument("--dataset", default="YOUR_DIR/data", type=str)
    parser.add_argument("--task", default="wmt", type=str)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument(
        "--selection_rate", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.8]
    )
    parser.add_argument(
        "--baseline", nargs="+", type=str, default=["Effort"], help="EL2N/CCS/Moderate/Effort"
    )
    args = parser.parse_args()

    save_dir = os.path.join(args.base_model, f"selected-{args.task}")
    os.makedirs(save_dir, exist_ok=True)

    scores_path = os.path.join(args.base_model, f"scores-{args.task}.pkl")
    feature_path = os.path.join(args.base_model, f"feature_{args.task}.pt")

    # 如果需要先计算特征和目标，再走 D2/Moderate 分支
    if ("D2" in args.baseline or "Moderate" in args.baseline) and not os.path.exists(feature_path):
        _ = get_baseline_score(
            base_model=args.base_model,
            dataset=args.dataset,
            tokenizer_path=args.tokenizer,
            finetune=args.finetune,
            method=args.baseline,
        )
        with open(scores_path, "rb") as f:
            result = pickle.load(f)
    else:
        if not os.path.exists(scores_path):
            result = get_baseline_score(
                base_model=args.base_model,
                dataset=args.dataset,
                tokenizer_path=args.tokenizer,
                finetune=args.fin
            )
            with open(scores_path, "wb") as f:
                pickle.dump(result, f)
        else:
            with open(scores_path, "rb") as f:
                result = pickle.load(f)

    # 其它后处理逻辑…
