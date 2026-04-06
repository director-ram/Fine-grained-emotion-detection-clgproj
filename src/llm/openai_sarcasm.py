from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OpenAISarcasmConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    timeout_s: float = 30.0
    base_url: str = "https://api.openai.com/v1"


def _join_url(base_url: str, path: str) -> str:
    # Ensure we get: <base>/... without double slashes.
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _extract_first_json_object(text: str) -> str | None:
    """
    Best-effort extraction when the model returns extra text.
    Looks for the first {...} block (non-nested heuristic).
    """
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None


class OpenAISarcasmClassifier:
    """
    Simple API-based sarcasm classifier using OpenAI Chat Completions.

    Returns:
      - label in {"sarcastic","non-sarcastic"}
      - score: 1.0 for sarcastic, 0.0 for non-sarcastic (not a calibrated probability)
    """

    def __init__(self, cfg: OpenAISarcasmConfig) -> None:
        self.cfg = cfg

    def predict(self, text: str) -> Tuple[str, float]:
        system = (
            "You are a strict sarcasm classifier.\n"
            "Given a single sentence, output ONLY a JSON object with keys:\n"
            '  {"label": "sarcastic"|"non-sarcastic"}\n'
            "No extra text."
        )
        user = f"Sentence: {text}"

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        req = urllib.request.Request(
            url=_join_url(self.cfg.base_url, "/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # LM Studio / local OpenAI-compatible servers may ignore this, but
                # OpenAI requires it.
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            detail = f"LLM HTTP {e.code}: {e.reason}"
            if body:
                detail = f"{detail} (body={body})"
            raise RuntimeError(detail) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM request failed: {e}") from e
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            extracted = _extract_first_json_object(content)
            if extracted is None:
                raise ValueError(f"LLM did not return JSON (raw={content!r})")
            parsed = json.loads(extracted)
        label = str(parsed.get("label", "")).strip().lower()
        if label not in {"sarcastic", "non-sarcastic"}:
            raise ValueError(f"Unexpected label from LLM: {label!r} (raw={content!r})")
        score = 1.0 if label == "sarcastic" else 0.0
        return label, score


def load_openai_classifier_from_env() -> OpenAISarcasmClassifier:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    # For local OpenAI-compatible servers (LM Studio), allow a dummy/empty key.
    if not api_key:
        if base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1"):
            api_key = "lm-studio"
        else:
            raise RuntimeError("OPENAI_API_KEY is not set.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    return OpenAISarcasmClassifier(
        OpenAISarcasmConfig(api_key=api_key, model=model, base_url=base_url)
    )

