from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GeminiSarcasmConfig:
    api_key: str
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    timeout_s: float = 30.0


class GeminiSarcasmClassifier:
    """
    Gemini API-based sarcasm classifier (REST generateContent).

    Returns:
      - label in {"sarcastic","non-sarcastic"}
      - score: 1.0 for sarcastic, 0.0 for non-sarcastic (not a calibrated probability)
    """

    def __init__(self, cfg: GeminiSarcasmConfig) -> None:
        self.cfg = cfg

    def predict(self, text: str) -> Tuple[str, float]:
        system = (
            "You are a strict sarcasm classifier.\n"
            "Given a single sentence, output ONLY a JSON object with keys:\n"
            '  {"label": "sarcastic"|"non-sarcastic"}\n'
            "No extra text."
        )
        prompt = f"{system}\n\nSentence: {text}"

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.cfg.model}:generateContent"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": float(self.cfg.temperature),
            },
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.cfg.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # Common causes: quota/rate limit (429), invalid key/permissions (401/403).
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            detail = f"Gemini API HTTP {e.code}: {e.reason}"
            if body:
                detail = f"{detail} (body={body})"
            raise RuntimeError(detail) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e
        data = json.loads(raw)

        # Typical response: candidates[0].content.parts[0].text
        content = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gemini did not return JSON (raw={content!r})") from e
        label = str(parsed.get("label", "")).strip().lower()
        if label not in {"sarcastic", "non-sarcastic"}:
            raise ValueError(f"Unexpected label from Gemini: {label!r} (raw={content!r})")
        score = 1.0 if label == "sarcastic" else 0.0
        return label, score


def load_gemini_classifier_from_env() -> GeminiSarcasmClassifier:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    return GeminiSarcasmClassifier(GeminiSarcasmConfig(api_key=api_key, model=model))

