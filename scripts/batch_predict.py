from pathlib import Path

import argparse
import sys

import pandas as pd  # pyright: ignore[reportMissingImports]

# Ensure the project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import InferenceConfig, SarcasmPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch sarcasm prediction over a CSV file."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/sarcasm_transformer_base/best_model"),
        help="Path to finetuned model directory.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to CSV file with a text column.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the text column in the CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5405,
        help=(
            "Probability threshold for labeling as sarcastic. "
            "sarcastic = (score >= threshold)."
        ),
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
    df = pd.read_csv(args.csv_path)

    if args.text_col not in df.columns:
        raise ValueError(
            f"Column '{args.text_col}' not found in {args.csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    for i, text in enumerate(df[args.text_col]):
        label, score = predictor.predict(str(text))
        is_sarcastic = score >= args.threshold
        print(f"{i}\t{is_sarcastic}\t{score:.4f}\t{text}")


if __name__ == "__main__":
    main()