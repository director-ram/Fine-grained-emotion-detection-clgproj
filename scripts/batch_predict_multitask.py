from __future__ import annotations

from pathlib import Path

import argparse
import sys

import pandas as pd  # pyright: ignore[reportMissingImports]
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import MultiTaskTweetDataset, build_tokenizer  # noqa: E402
from src.models.multitask_classifier import (  # noqa: E402
    MultiTaskConfig,
    MultiTaskSequenceClassifier,
)


def load_model_and_tokenizer(model_dir: Path) -> tuple[MultiTaskSequenceClassifier, any]:
    tokenizer = build_tokenizer(str(model_dir))
    cfg = MultiTaskConfig(pretrained_model_name_or_path=str(model_dir))
    model = MultiTaskSequenceClassifier(cfg)
    state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch multi-task sarcasm + emotion prediction.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained multi-task model directory.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to CSV file with a text column.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the text column.",
    )
    parser.add_argument(
        "--sarcasm-threshold",
        type=float,
        default=0.5,
        help="Threshold on sarcasm probability for True/False.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.csv_path}. Available: {list(df.columns)}")

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = MultiTaskTweetDataset(
        texts=df[args.text_col].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        sarcasm_labels=[None] * len(df),
        emotion_labels=[None] * len(df),
    )

    from torch.utils.data import DataLoader  # local import to avoid global dependency

    loader = DataLoader(dataset, batch_size=32)

    with torch.no_grad():
        idx_offset = 0
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            outputs = model(**inputs)
            logits_sarc = outputs["logits_sarcasm"]
            logits_emo = outputs["logits_emotion"]

            probs_sarc = logits_sarc.softmax(dim=-1)
            probs_emo = logits_emo.softmax(dim=-1)

            sarcasm_scores, sarcasm_pred_idx = probs_sarc.max(dim=-1)
            emotion_scores, emotion_pred_idx = probs_emo.max(dim=-1)

            for i in range(len(sarcasm_scores)):
                row_idx = idx_offset + i
                text = df[args.text_col].iloc[row_idx]
                sarc_score = float(sarcasm_scores[i])
                sarc_label = bool(sarc_score >= args.sarcasm_threshold)
                emo_id = int(emotion_pred_idx[i])
                print(f"{row_idx}\t{sarc_label}\t{sarc_score:.4f}\t{emo_id}\t{text}")

            idx_offset += len(sarcasm_scores)


if __name__ == "__main__":
    main()

