from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd  # pyright: ignore[reportMissingImports]
import torch
import tyro  # pyright: ignore[reportMissingImports]
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader

from src.data_loading import MultiTaskTweetDataset, build_tokenizer, load_emotion_csv, load_sarcasm_csv
from src.models.multitask_classifier import MultiTaskConfig, MultiTaskSequenceClassifier


@dataclass
class MultiTaskEvalConfig:
    model_dir: Path
    sarcasm_test_path: Path = Path("data/sarcasm/test.csv")
    emotion_test_path: Path = Path("data/emotion/test.csv")
    max_seq_length: int = 128
    batch_size: int = 64


def load_model_and_tokenizer(model_dir: Path) -> tuple[MultiTaskSequenceClassifier, any]:
    tokenizer = build_tokenizer(str(model_dir))
    # Load config to recover label sizes if needed
    base_config = MultiTaskConfig(
        pretrained_model_name_or_path=str(model_dir),
    )
    model = MultiTaskSequenceClassifier(base_config)
    state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    return model, tokenizer


def eval_sarcasm(cfg: MultiTaskEvalConfig, model: MultiTaskSequenceClassifier, tokenizer) -> Dict[str, float]:
    df = load_sarcasm_csv(cfg.sarcasm_test_path)
    dataset = MultiTaskTweetDataset(
        texts=df["text"].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        sarcasm_labels=df["sarcasm_label"].astype(int).tolist(),
        emotion_labels=[None] * len(df),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["sarcasm_labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("sarcasm_labels", "emotion_labels")}
            outputs = model(**inputs)
            logits = outputs["logits_sarcasm"]
            preds = logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    print("Sarcasm classification report:")
    print(classification_report(all_labels, all_preds, target_names=["non_sarcastic", "sarcastic"]))
    return {"sarcasm_accuracy": acc, "sarcasm_f1": f1}


def eval_emotion(cfg: MultiTaskEvalConfig, model: MultiTaskSequenceClassifier, tokenizer) -> Dict[str, float]:
    df = load_emotion_csv(cfg.emotion_test_path)
    dataset = MultiTaskTweetDataset(
        texts=df["text"].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        sarcasm_labels=[None] * len(df),
        emotion_labels=df["emotion_label"].astype(int).tolist(),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["emotion_labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("sarcasm_labels", "emotion_labels")}
            outputs = model(**inputs)
            logits = outputs["logits_emotion"]
            preds = logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("Emotion classification report (macro F1):")
    print(classification_report(all_labels, all_preds))
    return {"emotion_accuracy": acc, "emotion_macro_f1": macro_f1}


def main(cfg: MultiTaskEvalConfig) -> None:
    model, tokenizer = load_model_and_tokenizer(cfg.model_dir)
    sarcasm_metrics = eval_sarcasm(cfg, model, tokenizer)
    emotion_metrics = eval_emotion(cfg, model, tokenizer)
    print("Sarcasm metrics:", sarcasm_metrics)
    print("Emotion metrics:", emotion_metrics)


if __name__ == "__main__":
    tyro.cli(main)

