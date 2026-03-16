"""
Minimal script to create a tiny demo sarcasm dataset for development and testing.

In a real project you would replace this with logic that downloads and normalizes
one or more real sarcasm corpora into the unified schema (text, label).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import DataPaths, save_splits, train_val_test_split
from src.preprocessing import standardize_schema


def build_demo_dataframe() -> pd.DataFrame:
    examples = [
        # non-sarcastic (0)
        ("I really enjoyed this movie.", 0),
        ("The food was delicious and the service was great.", 0),
        ("Thank you for your help today.", 0),
        ("It is raining outside.", 0),
        ("She finished her work on time.", 0),
        # sarcastic (1)
        ("Oh great, another meeting that could have been an email.", 1),
        ("Yeah, because waiting in traffic is my favorite hobby.", 1),
        ("Wonderful, my phone died right before the call.", 1),
        ("Fantastic, the printer stopped working again.", 1),
        ("Just what I needed, more bugs in production.", 1),
    ]
    df = pd.DataFrame(examples, columns=["sentence", "is_sarcastic"])
    return df


def main() -> None:
    data_paths = DataPaths()
    raw_dir = data_paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    df_raw = build_demo_dataframe()
    df_raw.to_csv(raw_dir / "demo_sarcasm.csv", index=False)

    df = standardize_schema(df_raw, text_col="sentence", label_col="is_sarcastic")
    train_df, val_df, test_df = train_val_test_split(df, val_size=0.2, test_size=0.2)
    save_splits(train_df, val_df, test_df, processed_dir=data_paths.processed_dir)

    print("Demo sarcasm dataset prepared under data/processed/")


if __name__ == "__main__":
    main()

