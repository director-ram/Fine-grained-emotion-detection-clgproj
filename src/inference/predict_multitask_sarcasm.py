from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer

from src.models.multitask_classifier import MultiTaskConfig, MultiTaskSequenceClassifier


@dataclass
class MultiTaskSarcasmInferenceConfig:
    model_dir: Path
    max_seq_length: int | None = None
    threshold: float | None = None


def _load_multitask_cfg(model_dir: Path) -> MultiTaskConfig:
    cfg_path = model_dir / "multitask_config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        return MultiTaskConfig(
            pretrained_model_name_or_path=str(raw["pretrained_model_name_or_path"]),
            num_sarcasm_labels=int(raw.get("num_sarcasm_labels", 2)),
            num_emotion_labels=int(raw.get("num_emotion_labels", 6)),
            lambda_sarcasm=float(raw.get("lambda_sarcasm", 1.0)),
            lambda_emotion=float(raw.get("lambda_emotion", 1.0)),
        )
    return MultiTaskConfig(pretrained_model_name_or_path="bert-base-uncased")


def _load_state_dict(model_dir: Path) -> dict:
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file as safetensors_load_file  # pyright: ignore[reportMissingImports]

        return safetensors_load_file(str(safetensors_path))
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(
        f"Could not find model weights in {model_dir}. "
        f"Expected either '{safetensors_path.name}' or '{bin_path.name}'."
    )


class MultiTaskSarcasmPredictor:
    def __init__(self, cfg: MultiTaskSarcasmInferenceConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disk_cfg: dict = {}
        inference_cfg_path = cfg.model_dir / "inference_config.json"
        if inference_cfg_path.exists():
            disk_cfg = json.loads(inference_cfg_path.read_text(encoding="utf-8"))
        self.max_seq_length = int(
            cfg.max_seq_length
            if cfg.max_seq_length is not None
            else disk_cfg.get("max_seq_length", 64)
        )
        self.threshold = float(
            cfg.threshold if cfg.threshold is not None else disk_cfg.get("threshold", 0.5)
        )
        mt_cfg = _load_multitask_cfg(cfg.model_dir)
        self.model = MultiTaskSequenceClassifier(mt_cfg)
        state = _load_state_dict(cfg.model_dir)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_dir), use_fast=True)

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        logits = outputs["logits_sarcasm"].squeeze(0)
        probs = logits.softmax(dim=-1)
        sarcastic_prob = float(probs[1].item())
        label = "sarcastic" if sarcastic_prob >= float(self.threshold) else "non-sarcastic"
        return label, sarcastic_prob


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-task sarcasm (sarcasm head) CLI")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Sarcasm probability threshold (default: from inference_config.json, fallback 0.5).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    predictor = MultiTaskSarcasmPredictor(
        MultiTaskSarcasmInferenceConfig(
            model_dir=args.model_dir,
            max_seq_length=args.max_seq_length,
            threshold=args.threshold,
        )
    )
    label, score = predictor.predict(args.text)
    message = "yes, its sarcastic" if label == "sarcastic" else "no, its not sarcastic"
    if args.verbose:
        print(f"Text: {args.text}")
        print(f"Prediction: {label} (score={score:.4f})")
    print(message)


if __name__ == "__main__":
    _cli()

