from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import argparse
import pandas as pd  # pyright: ignore[reportMissingImports]
import json
import torch
from safetensors.torch import load_file as safetensors_load_file  # pyright: ignore[reportMissingImports]
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from src.data_loading import (
    MultiTaskTweetDataset,
    build_tokenizer,
    load_emotion_csv,
    load_sarcasm_csv,
)
from src.models.multitask_classifier import MultiTaskConfig, MultiTaskSequenceClassifier


@dataclass
class MultiTaskEvalConfig:
    model_dir: Path
    sarcasm_test_path: Path = Path("data/sarcasm/test.csv")
    emotion_test_path: Path = Path("data/emotion/test.csv")
    max_seq_length: int = 128
    batch_size: int = 64


def _load_state_dict(model_dir: Path) -> dict:
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        return safetensors_load_file(str(safetensors_path))

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"Could not find model weights in {model_dir}. "
        f"Expected either '{safetensors_path.name}' or '{bin_path.name}'."
    )


def load_model_and_tokenizer(model_dir: Path) -> tuple[MultiTaskSequenceClassifier, any]:
    tokenizer = build_tokenizer(str(model_dir))
    multitask_cfg_path = model_dir / "multitask_config.json"
    if multitask_cfg_path.exists():
        multitask_cfg = json.loads(multitask_cfg_path.read_text(encoding="utf-8"))
        cfg = MultiTaskConfig(
            pretrained_model_name_or_path=str(multitask_cfg["pretrained_model_name_or_path"]),
            num_sarcasm_labels=int(multitask_cfg.get("num_sarcasm_labels", 2)),
            num_emotion_labels=int(multitask_cfg.get("num_emotion_labels", 6)),
            lambda_sarcasm=float(multitask_cfg.get("lambda_sarcasm", 1.0)),
            lambda_emotion=float(multitask_cfg.get("lambda_emotion", 1.0)),
        )
    else:
        # Best-effort fallback (older checkpoints); user can still re-train to generate multitask_config.json.
        cfg = MultiTaskConfig(pretrained_model_name_or_path="bert-base-uncased")

    model = MultiTaskSequenceClassifier(cfg)
    state_dict = _load_state_dict(model_dir)
    model.load_state_dict(state_dict, strict=False)
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=data_collator)

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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=data_collator)

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
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-task sarcasm + emotion model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained multi-task model directory (e.g. outputs/.../best_model_multitask).",
    )
    parser.add_argument(
        "--sarcasm-test-path",
        type=Path,
        default=Path("data/sarcasm/test.csv"),
    )
    parser.add_argument(
        "--emotion-test-path",
        type=Path,
        default=Path("data/emotion/test.csv"),
    )
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    main(
        MultiTaskEvalConfig(
            model_dir=args.model_dir,
            sarcasm_test_path=args.sarcasm_test_path,
            emotion_test_path=args.emotion_test_path,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
    )

