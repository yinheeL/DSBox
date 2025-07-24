
import random
import sys
import os
import argparse
import logging
import multiprocessing
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from sklearn.metrics import accuracy_score, f1_score


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from selectiontools.selection_core import selection
from selectiontools.dataset import DevignDataset
from selectiontools.model_interface import DevignCodeBERTModel

# ACC&F1
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}


def parse_args():
    parser = argparse.ArgumentParser(description='Train Devign+CodeBERT with data selection')
    parser.add_argument('--data_dir', type=str, default='selectiontools/data/devign', help='data directory')
    parser.add_argument('--metric', type=str, required=True, help='strategies')
    parser.add_argument('--budget', type=float, required=True, help='budget<=1&&budget>0')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='the ratio of evaluation data')
    parser.add_argument('--output_dir', type=str, required=True, help='save')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    # dataset
    raw_ds = DevignDataset(args.data_dir)
    N = len(raw_ds)
    idxs = list(range(N))
    random.shuffle(idxs)
    split = int(N * (1 - args.eval_ratio))
    train_idx, eval_idx = idxs[:split], idxs[split:]
    train_ds = DevignDataset(args.data_dir, shuffle=False)
    train_ds.unlabeled_indices = train_idx
    eval_ds = DevignDataset(args.data_dir, shuffle=False)
    eval_ds.unlabeled_indices = eval_idx

    # select
    B = int(args.budget * len(train_idx)) if args.budget <= 1 else int(args.budget)
    logger.warning(f'Selecting {B}/{len(train_idx)} samples with {args.metric}')
    model_if = DevignCodeBERTModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    selected = selection(model_if, train_ds, args.metric, B)
    train_ds.unlabeled_indices = selected

    # preparation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DevignCodeBERTModel(device=str(device))
    model.to(device)
    bs, epochs = 32, 10
    steps_per_epoch = max(1, len(train_ds) // bs)
    logger.warning(f'Training on {len(train_ds)} samples: bs={bs}, epochs={epochs}, steps/epoch={steps_per_epoch}')

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        save_steps=0,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    # train&evaluation
    trainer.train()
    eval_metrics = trainer.evaluate()
    final_acc = eval_metrics.get('eval_accuracy')
    final_f1 = eval_metrics.get('eval_f1')

    # save
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Accuracy: {final_acc:.4f}\n')
        f.write(f'F1 Score: {final_f1:.4f}\n')
    logger.warning(f'Saved final metrics to {metrics_path}')
    print(f'Saved metrics to {metrics_path}', flush=True)
