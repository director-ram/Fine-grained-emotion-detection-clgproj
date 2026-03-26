from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import tyro
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import DataCollatorWithPadding

from src.data_loading import build_tokenizer, tokenize_dataframe
from src.models.sarcasm_classifier import load_finetuned_model


@dataclass
class EvalConfig:
    model_dir: Path
    test_path: Path = Path("data/processed/test.csv")
    max_seq_length: int = 128


def evaluate(cfg: EvalConfig) -> Dict[str, float]:
    model = load_finetuned_model(str(cfg.model_dir))
    tokenizer = build_tokenizer(str(cfg.model_dir))

    test_df = pd.read_csv(cfg.test_path)
    test_dataset = tokenize_dataframe(test_df, tokenizer, cfg.max_seq_length)

    # Manually run inference to have full control over metrics and reports
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["non-sarcastic", "sarcastic"],
        )
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    print("\nMetrics:", metrics)
    return metrics


def main(cfg: EvalConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    tyro.cli(main)

