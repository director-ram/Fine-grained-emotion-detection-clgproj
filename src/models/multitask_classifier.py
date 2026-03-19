from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
)


@dataclass
class MultiTaskConfig:
    pretrained_model_name_or_path: str
    num_sarcasm_labels: int = 2
    num_emotion_labels: int = 6
    lambda_sarcasm: float = 1.0
    lambda_emotion: float = 1.0


class MultiTaskSequenceClassifier(PreTrainedModel):
    """
    Shared encoder with two classification heads:
    - sarcasm_head: binary sarcasm / irony
    - emotion_head: multi-class emotion
    """

    config_class = AutoConfig

    def __init__(self, cfg: MultiTaskConfig) -> None:
        base_config = AutoConfig.from_pretrained(
            cfg.pretrained_model_name_or_path,
        )
        super().__init__(base_config)

        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(
            cfg.pretrained_model_name_or_path, config=base_config
        )
        hidden_size = base_config.hidden_size

        self.sarcasm_head = nn.Linear(hidden_size, cfg.num_sarcasm_labels)
        self.emotion_head = nn.Linear(hidden_size, cfg.num_emotion_labels)

        self.loss_sarcasm = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_emotion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        sarcasm_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # Use pooled output when available, else CLS token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        logits_sarc = self.sarcasm_head(pooled)
        logits_emo = self.emotion_head(pooled)

        loss = None
        loss_sarc_val: Optional[torch.Tensor] = None
        loss_emo_val: Optional[torch.Tensor] = None

        if sarcasm_labels is not None:
            # CrossEntropyLoss(ignore_index=-100) returns NaN if *all* targets are ignored.
            if torch.any(sarcasm_labels != -100):
                loss_sarc_val = self.loss_sarcasm(logits_sarc, sarcasm_labels)
        if emotion_labels is not None:
            if torch.any(emotion_labels != -100):
                loss_emo_val = self.loss_emotion(logits_emo, emotion_labels)

        if loss_sarc_val is not None or loss_emo_val is not None:
            loss = torch.zeros((), device=logits_sarc.device)
            if loss_sarc_val is not None:
                loss = loss + self.cfg.lambda_sarcasm * loss_sarc_val
            if loss_emo_val is not None:
                loss = loss + self.cfg.lambda_emotion * loss_emo_val

        return {
            "loss": loss,
            "logits_sarcasm": logits_sarc,
            "logits_emotion": logits_emo,
        }


def build_multitask_model(cfg: MultiTaskConfig) -> MultiTaskSequenceClassifier:
    return MultiTaskSequenceClassifier(cfg)

