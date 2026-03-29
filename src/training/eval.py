from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import pandas as pd  # pyright: ignore[reportMissingImports]
import tyro  # pyright: ignore[reportMissingImports]
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    accuracy_score,
    balanced_accuracy_score,
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
    val_path: Path = Path("data/processed/val.csv")
    max_seq_length: int = 128
    optimize_threshold: bool = True
    threshold_metric: str = "f1"
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_step: float = 0.05
    save_threshold: bool = True


def _sweep_threshold(
    labels: list[int],
    sarcastic_probs: list[float],
    metric: str,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[float, float]:
    if metric not in {"f1", "balanced_accuracy"}:
        raise ValueError("threshold_metric must be one of {'f1', 'balanced_accuracy'}")
    best_t = 0.5
    best_score = -1.0
    t = t_min
    while t <= t_max + 1e-9:
        preds = [1 if p >= t else 0 for p in sarcastic_probs]
        score = (
            f1_score(labels, preds, average="binary")
            if metric == "f1"
            else balanced_accuracy_score(labels, preds)
        )
        if score > best_score:
            best_score = float(score)
            best_t = float(round(t, 4))
        t += t_step
    return best_t, best_score


def _run_inference(model, tokenizer, csv_path: Path, max_seq_length: int) -> tuple[list[int], list[int], list[float]]:
    import torch
    from torch.utils.data import DataLoader

    df = pd.read_csv(csv_path)
    dataset = tokenize_dataframe(df, tokenizer, max_seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=32, collate_fn=data_collator)

    all_labels: list[int] = []
    all_preds: list[int] = []
    sarcastic_probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = outputs.logits.softmax(dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            sarcastic_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, sarcastic_probs


def evaluate(cfg: EvalConfig) -> Dict[str, float]:
    model = load_finetuned_model(str(cfg.model_dir))
    tokenizer = build_tokenizer(str(cfg.model_dir))
    threshold = 0.5
    threshold_source = "default"

    if cfg.optimize_threshold and cfg.val_path.exists():
        val_labels, _, val_probs = _run_inference(
            model=model,
            tokenizer=tokenizer,
            csv_path=cfg.val_path,
            max_seq_length=cfg.max_seq_length,
        )
        metric_name = (
            "balanced_accuracy"
            if cfg.threshold_metric == "balanced_accuracy"
            else "f1"
        )
        threshold, threshold_score = _sweep_threshold(
            labels=val_labels,
            sarcastic_probs=val_probs,
            metric=metric_name,
            t_min=cfg.threshold_min,
            t_max=cfg.threshold_max,
            t_step=cfg.threshold_step,
        )
        threshold_source = f"val:{cfg.val_path.name}:{metric_name}"
        print(
            f"Selected threshold={threshold:.4f} using {metric_name} on val "
            f"(score={threshold_score:.4f})"
        )

    all_labels, _, all_probs = _run_inference(
        model=model,
        tokenizer=tokenizer,
        csv_path=cfg.test_path,
        max_seq_length=cfg.max_seq_length,
    )
    all_preds = [1 if p >= threshold else 0 for p in all_probs]

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    pos_rate = sum(all_preds) / max(1, len(all_preds))

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
        "threshold": threshold,
        "predicted_positive_rate": pos_rate,
    }
    print("\nMetrics:", metrics)
    if pos_rate <= 0.05 or pos_rate >= 0.95:
        print(
            "WARNING: prediction collapse detected "
            f"(predicted_positive_rate={pos_rate:.3f})."
        )

    if cfg.save_threshold:
        inference_cfg_path = cfg.model_dir / "inference_config.json"
        payload: Dict[str, object] = {}
        if inference_cfg_path.exists():
            with inference_cfg_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        payload["threshold"] = float(threshold)
        payload["threshold_source"] = threshold_source
        payload.setdefault("max_seq_length", cfg.max_seq_length)
        with inference_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved calibrated threshold to {inference_cfg_path}")
    return metrics


def main(cfg: EvalConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    tyro.cli(main)

