from __future__ import annotations

"""
Download and normalize public Twitter datasets for sarcasm/irony and emotion.

Outputs:
- data/sarcasm/train.csv, val.csv, test.csv   (columns: text, sarcasm_label)
- data/emotion/train.csv, val.csv, test.csv   (columns: text, emotion_label)
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def prepare_sarcasm(output_dir: Path) -> None:
    ds_irony = load_dataset("S-a-r-a/tweet_eval", "irony")

    def to_df(split) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": split["text"],
                "sarcasm_label": split["label"],  # 0 = non_irony, 1 = irony
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    to_df(ds_irony["train"]).to_csv(output_dir / "train.csv", index=False)
    to_df(ds_irony["validation"]).to_csv(output_dir / "val.csv", index=False)
    to_df(ds_irony["test"]).to_csv(output_dir / "test.csv", index=False)


def prepare_emotion(output_dir: Path) -> None:
    ds_emo = load_dataset("dair-ai/emotion")
    label_names = ds_emo["train"].features["label"].names
    print("Emotion labels:", label_names)

    def to_df(split) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": split["text"],
                "emotion_label": split["label"],  # int ids into label_names
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    to_df(ds_emo["train"]).to_csv(output_dir / "train.csv", index=False)
    to_df(ds_emo["validation"]).to_csv(output_dir / "val.csv", index=False)
    to_df(ds_emo["test"]).to_csv(output_dir / "test.csv", index=False)


def main() -> None:
    base = Path("data")
    prepare_sarcasm(base / "sarcasm")
    prepare_emotion(base / "emotion")
    print("Finished downloading sarcasm and emotion datasets into 'data/'.")


if __name__ == "__main__":
    main()

