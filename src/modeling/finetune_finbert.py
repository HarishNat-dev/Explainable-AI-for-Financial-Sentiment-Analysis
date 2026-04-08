from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "ProsusAI/finbert"

    train_path: str = "data/processed/phrasebank_allagree_train.parquet"
    val_path: str = "data/processed/phrasebank_allagree_val.parquet"
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"

    out_dir: str = "models/finbert/finbert_phrasebank_allagree"
    metrics_dir: str = "reports/metrics"

    seed: int = 42
    max_length: int = 128
    lr: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 4
    batch_size: int = 8
    eval_steps: int = 100
    patience: int = 2


class WeightedTrainer(Trainer):
    """Trainer that applies class-weighted cross entropy to handle imbalance."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
    }


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["sentence", "label"]], preserve_index=False)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    metrics_dir = Path(cfg.metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(cfg.train_path)
    val_df = pd.read_parquet(cfg.val_path)
    test_df = pd.read_parquet(cfg.test_path)

    classes = np.array(sorted(train_df["label"].unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = to_hf_dataset(train_df).map(tokenize, batched=True)
    val_ds = to_hf_dataset(val_df).map(tokenize, batched=True)
    test_ds = to_hf_dataset(test_df).map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    print("\nTest metrics:")
    print(test_metrics)

    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    metrics_path = metrics_dir / "finbert_phrasebank_allagree_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
