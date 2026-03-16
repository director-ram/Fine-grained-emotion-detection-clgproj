from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from src.data_loading import build_tokenizer
from src.models.sarcasm_classifier import load_finetuned_model


LABELS = ["non-sarcastic", "sarcastic"]


@dataclass
class InferenceConfig:
    model_dir: Path
    max_seq_length: int = 128


class SarcasmPredictor:
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_finetuned_model(str(cfg.model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = build_tokenizer(str(cfg.model_dir))

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        probs = outputs.logits.softmax(dim=-1).squeeze(0)
        score, label_idx = torch.max(probs, dim=-1)
        label_name = LABELS[int(label_idx)]
        return label_name, float(score)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sarcasm detection CLI")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to finetuned model directory (e.g. outputs/exp/best_model)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input sentence to classify.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    args = parser.parse_args()

    predictor = SarcasmPredictor(
        InferenceConfig(model_dir=args.model_dir, max_seq_length=args.max_seq_length)
    )
    label, score = predictor.predict(args.text)
    print(f"Text: {args.text}")
    print(f"Prediction: {label} (score={score:.4f})")


if __name__ == "__main__":
    _cli()

