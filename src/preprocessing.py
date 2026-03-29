from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd  # pyright: ignore[reportMissingImports]


@dataclass
class TextColumns:
    text: str = "text"
    label: str = "label"


def basic_clean_text(series: pd.Series) -> pd.Series:
    """Light text normalization suitable for transformer models."""
    # We intentionally avoid heavy normalization since modern models handle noise well.
    series = series.astype(str)
    series = series.str.replace("\r\n", " ", regex=False)
    series = series.str.replace("\n", " ", regex=False)
    series = series.str.replace("\t", " ", regex=False)
    series = series.str.replace(r"\s+", " ", regex=True)
    return series.str.strip()


def standardize_schema(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Return a dataframe with canonical columns: text, label."""
    missing: Iterable[str] = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    out = pd.DataFrame(
        {
            "text": df[text_col],
            "label": df[label_col].astype(int),
        }
    )
    out["text"] = basic_clean_text(out["text"])
    return out

