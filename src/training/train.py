from __future__ import annotations

import random
from dataclasses import dataclass
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
        seed=seed,
        report_to=[],  # disable default reporting integrations (e.g., wandb) by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    print(f"Training complete. Best model saved to {output_dir / 'best_model'}")


if __name__ == "__main__":
    tyro.cli(main)

