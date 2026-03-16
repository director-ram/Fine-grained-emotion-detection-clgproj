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
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

from src.data_loading import (
    MultiTaskTweetDataset,
    build_tokenizer,
    load_emotion_csv,
    load_sarcasm_csv,
)
from src.models.multitask_classifier import MultiTaskConfig, build_multitask_model


@dataclass
class MultiTaskTrainConfig:
    config_path: Path = Path("configs/multitask.yaml")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def main(cfg: MultiTaskTrainConfig) -> None:
    raw_cfg = load_yaml_config(cfg.config_path)

    seed = int(raw_cfg.get("seed", 42))
    set_global_seeds(seed)

    data_cfg = raw_cfg["data"]
    model_cfg = raw_cfg["model"]
    train_cfg = raw_cfg["training"]
    logging_cfg = raw_cfg["logging"]

    sarcasm_train = load_sarcasm_csv(Path(data_cfg["sarcasm_train_path"]))
    sarcasm_val = load_sarcasm_csv(Path(data_cfg["sarcasm_val_path"]))
    emotion_train = load_emotion_csv(Path(data_cfg["emotion_train_path"]))
    emotion_val = load_emotion_csv(Path(data_cfg["emotion_val_path"]))

    tokenizer = build_tokenizer(model_cfg["pretrained_model_name_or_path"])
    max_seq_length = int(train_cfg.get("max_seq_length", 128))

    # Build combined multi-task datasets
    train_texts = (
        sarcasm_train["text"].tolist() + emotion_train["text"].tolist()
    )
    train_sarc_labels = (
        sarcasm_train["sarcasm_label"].astype(int).tolist()
        + [None] * len(emotion_train)
    )
    train_emo_labels = (
        [None] * len(sarcasm_train)
        + emotion_train["emotion_label"].astype(int).tolist()
    )

    val_texts = sarcasm_val["text"].tolist() + emotion_val["text"].tolist()
    val_sarc_labels = (
        sarcasm_val["sarcasm_label"].astype(int).tolist()
        + [None] * len(emotion_val)
    )
    val_emo_labels = (
        [None] * len(sarcasm_val)
        + emotion_val["emotion_label"].astype(int).tolist()
    )

    train_dataset = MultiTaskTweetDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        sarcasm_labels=train_sarc_labels,
        emotion_labels=train_emo_labels,
    )
    val_dataset = MultiTaskTweetDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        sarcasm_labels=val_sarc_labels,
        emotion_labels=val_emo_labels,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = build_multitask_model(
        MultiTaskConfig(
            pretrained_model_name_or_path=model_cfg["pretrained_model_name_or_path"],
            num_sarcasm_labels=int(model_cfg.get("num_sarcasm_labels", 2)),
            num_emotion_labels=int(model_cfg.get("num_emotion_labels", 6)),
            lambda_sarcasm=float(model_cfg.get("lambda_sarcasm", 1.0)),
            lambda_emotion=float(model_cfg.get("lambda_emotion", 1.0)),
        )
    )

    output_dir = Path(logging_cfg["output_dir"]) / raw_cfg.get(
        "experiment_name", "multitask_experiment"
    )
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
        metric_for_best_model="loss",
        logging_steps=int(logging_cfg.get("log_steps", 100)),
        save_total_limit=int(logging_cfg.get("save_total_limit", 3)),
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    best_model_dir = output_dir / "best_model_multitask"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    print(f"Multi-task training complete. Best model saved to {best_model_dir}")


if __name__ == "__main__":
    tyro.cli(main)

