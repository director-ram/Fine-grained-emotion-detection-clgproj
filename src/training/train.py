from __future__ import annotations

import random
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]
import torch
import tyro  # pyright: ignore[reportMissingImports]
import yaml
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

from src.data_loading import build_tokenizer, tokenize_dataframe
from src.models.sarcasm_classifier import SarcasmModelConfig, build_sarcasm_model


@dataclass
class TrainConfig:
    config_path: Path = Path("configs/base.yaml")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary")
    rec = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted CrossEntropy loss for imbalance handling."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = None
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if return_outputs:
            return loss, outputs
        return loss


def compute_balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    """Return inverse-frequency normalized weights for classes 0..N-1."""
    counts = np.bincount(labels.astype(int))
    if counts.size < 2:
        raise ValueError("Expected at least two classes for sarcasm training.")
    if np.any(counts == 0):
        raise ValueError(
            f"One or more classes missing from train split. Counts={counts.tolist()}."
        )
    num_classes = float(counts.size)
    total = float(labels.shape[0])
    weights = total / (num_classes * counts.astype(np.float64))
    # Normalize for stable scale while preserving relative weighting.
    weights = weights / weights.mean()
    return weights


def main(cfg: TrainConfig) -> None:
    raw_cfg = load_yaml_config(cfg.config_path)

    seed = int(raw_cfg.get("seed", 42))
    set_global_seeds(seed)

    data_cfg = raw_cfg["data"]
    model_cfg = raw_cfg["model"]
    train_cfg = raw_cfg["training"]
    logging_cfg = raw_cfg["logging"]

    train_df = pd.read_csv(data_cfg["train_path"])
    val_df = pd.read_csv(data_cfg["val_path"])
    test_path = Path(str(data_cfg.get("test_path", "data/processed/test.csv")))

    # Validate labels and show split balance before tokenization.
    for split_name, split_df in (("train", train_df), ("val", val_df)):
        if "label" not in split_df.columns:
            raise ValueError(f"{split_name} split must contain a 'label' column.")
        unique_labels = set(split_df["label"].astype(int).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"{split_name} split labels must be binary 0/1. Got: {sorted(unique_labels)}"
            )
    train_counts = train_df["label"].astype(int).value_counts().sort_index().to_dict()
    val_counts = val_df["label"].astype(int).value_counts().sort_index().to_dict()
    print(f"Label distribution train={train_counts} val={val_counts}")

    tokenizer = build_tokenizer(model_cfg["pretrained_model_name_or_path"])
    max_seq_length = int(train_cfg.get("max_seq_length", 128))

    train_dataset = tokenize_dataframe(train_df, tokenizer, max_seq_length)
    val_dataset = tokenize_dataframe(val_df, tokenizer, max_seq_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = build_sarcasm_model(
        SarcasmModelConfig(
            pretrained_model_name_or_path=model_cfg["pretrained_model_name_or_path"],
            num_labels=int(model_cfg.get("num_labels", 2)),
        )
    )

    output_dir = Path(logging_cfg["output_dir"]) / raw_cfg.get("experiment_name", "experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = float(train_cfg["num_epochs"])
    batch_size = int(train_cfg["batch_size"])
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    total_steps = max(1, int(len(train_dataset) * num_epochs / batch_size))
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=int(logging_cfg.get("log_steps", 50)),
        save_total_limit=int(logging_cfg.get("save_total_limit", 3)),
        # Windows can intermittently fail safetensors writes due to mapped file locks.
        save_safetensors=False,
        seed=seed,
        report_to=[],  # disable default reporting integrations (e.g., wandb) by default
    )

    class_weights_np = compute_balanced_class_weights(train_df["label"].to_numpy())
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    inference_meta = {
        "max_seq_length": max_seq_length,
        "threshold": 0.5,
        "threshold_source": "default",
        "class_weights": class_weights_np.tolist(),
        "label_mapping": {"0": "non-sarcastic", "1": "sarcastic"},
        "data_paths": {
            "train": data_cfg["train_path"],
            "val": data_cfg["val_path"],
            "test": str(test_path),
        },
    }
    with (output_dir / "best_model" / "inference_config.json").open("w", encoding="utf-8") as f:
        json.dump(inference_meta, f, indent=2)

    print(f"Training complete. Best model saved to {output_dir / 'best_model'}")


if __name__ == "__main__":
    tyro.cli(main)

