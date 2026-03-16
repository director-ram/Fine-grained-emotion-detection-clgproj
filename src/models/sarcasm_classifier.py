from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)


@dataclass
class SarcasmModelConfig:
    pretrained_model_name_or_path: str
    num_labels: int = 2


def build_sarcasm_model(cfg: SarcasmModelConfig) -> PreTrainedModel:
    """Create a transformer-based sequence classifier for sarcasm detection."""
    hf_config = AutoConfig.from_pretrained(
        cfg.pretrained_model_name_or_path,
        num_labels=cfg.num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_model_name_or_path,
        config=hf_config,
    )
    return model


def load_finetuned_model(model_dir: str, num_labels: Optional[int] = None) -> PreTrainedModel:
    """Load a finetuned sarcasm classification model from disk."""
    hf_config = AutoConfig.from_pretrained(model_dir)
    if num_labels is not None:
        hf_config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=hf_config,
    )
    return model

