from __future__ import annotations

"""
Calibrate a probability threshold for the multi-task sarcasm head using a validation set.

Prereq:
  python scripts/download_public_datasets.py

Writes:
  outputs/multitask_tweets/best_model_multitask/inference_config.json
or any --model-dir you provide.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from src.inference.predict_multitask_sarcasm import _load_multitask_cfg, _load_state_dict
from src.models.multitask_classifier import MultiTaskSequenceClassifier


def sweep_threshold(y_true: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (p1 >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, average="binary"))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate multitask sarcasm threshold.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs") / "multitask_tweets" / "best_model_multitask",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data") / "sarcasm" / "val.csv",
        help="CSV with columns: text, sarcasm_label",
    )
    parser.add_argument("--max-seq-length", type=int, default=64)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"Model dir not found: {args.model_dir}")
    if not args.val_csv.exists():
        raise SystemExit(
            f"Validation CSV not found: {args.val_csv}. Run scripts/download_public_datasets.py first."
        )

    df = pd.read_csv(args.val_csv)
    if "text" not in df.columns or "sarcasm_label" not in df.columns:
        raise SystemExit(
            f"Expected columns ['text','sarcasm_label'] in {args.val_csv}, got {list(df.columns)}"
        )

    y_true = df["sarcasm_label"].astype(int).to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True)
    cfg = _load_multitask_cfg(args.model_dir)
    model = MultiTaskSequenceClassifier(cfg)
    model.load_state_dict(_load_state_dict(args.model_dir), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    probs_1: list[float] = []
    batch_size = 64
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out["logits_sarcasm"]
        p1 = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        probs_1.extend(p1.tolist())

    probs_1_np = np.asarray(probs_1, dtype=np.float64)
    best_t, best_f1 = sweep_threshold(y_true, probs_1_np)

    out_path = args.model_dir / "inference_config.json"
    payload = {"max_seq_length": int(args.max_seq_length), "threshold": float(best_t), "threshold_source": "val_sweep_f1"}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Calibrated threshold={best_t:.3f} (val_f1={best_f1:.4f})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

