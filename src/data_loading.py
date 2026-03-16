from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # pyright: ignore[reportMissingImports]
import torch
from sklearn.model_selection import StratifiedShuffleSplit  # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .preprocessing import standardize_schema


@dataclass
class DataPaths:
    raw_dir: Path = Path("data") / "raw"
    processed_dir: Path = Path("data") / "processed"


def load_raw_csv(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    df = pd.read_csv(path)
    return standardize_schema(df, text_column, label_column)


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split on label column."""
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in (0,1) and sum to < 1.")

    labels = df["label"]

    first_split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(first_split.split(df, labels))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    remaining = 1.0 - test_size
    val_rel_size = val_size / remaining

    second_split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_rel_size, random_state=random_state
    )
    train_idx, val_idx = next(second_split.split(train_val_df, train_val_df["label"]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path | None = None,
) -> None:
    processed_dir = processed_dir or DataPaths().processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)


class SarcasmDataset(Dataset):
    """Torch dataset built from tokenized encodings and labels."""

    def __init__(self, encodings: Dict[str, List[int]], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer for the given pretrained model name."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> SarcasmDataset:
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=max_length,
    )
    labels: List[int] = df["label"].astype(int).tolist()
    return SarcasmDataset(encodings, labels)


def load_sarcasm_csv(path: Path) -> pd.DataFrame:
    """Load sarcasm CSV with columns: text, sarcasm_label."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "sarcasm_label" not in df.columns:
        raise ValueError(f"Expected columns ['text', 'sarcasm_label'] in {path}, got {list(df.columns)}")
    return df


def load_emotion_csv(path: Path) -> pd.DataFrame:
    """Load emotion CSV with columns: text, emotion_label."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "emotion_label" not in df.columns:
        raise ValueError(f"Expected columns ['text', 'emotion_label'] in {path}, got {list(df.columns)}")
    return df


class MultiTaskTweetDataset(Dataset):
    """
    Dataset for multi-task training.
    Each example may have:
    - sarcasm_label in {0,1} or None
    - emotion_label in {0..num_emotions-1} or None
    Labels that are None are encoded as -100 for ignored loss entries.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        sarcasm_labels: Optional[List[Optional[int]]] = None,
        emotion_labels: Optional[List[Optional[int]]] = None,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
        )
        n = len(texts)
        if sarcasm_labels is None:
            sarcasm_labels = [None] * n
        if emotion_labels is None:
            emotion_labels = [None] * n
        self.sarcasm_labels = [
            (-100 if lbl is None else int(lbl)) for lbl in sarcasm_labels
        ]
        self.emotion_labels = [
            (-100 if lbl is None else int(lbl)) for lbl in emotion_labels
        ]

    def __len__(self) -> int:
        return len(self.sarcasm_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["sarcasm_labels"] = torch.tensor(self.sarcasm_labels[idx])
        item["emotion_labels"] = torch.tensor(self.emotion_labels[idx])
        return item



