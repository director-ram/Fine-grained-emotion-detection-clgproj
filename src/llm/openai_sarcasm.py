from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Tuple

from src.llm.sarcasm_llm_prompt import SARCASM_SYSTEM_PROMPT


@dataclass
class OpenAISarcasmConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    timeout_s: float = 30.0
    base_url: str = "https://api.openai.com/v1"
    max_retries: int = 1


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
        system = SARCASM_SYSTEM_PROMPT
        user = f"Sentence: {text}\n\nIf unsure, respond with label non-sarcastic."

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        # Fast reachability check (helps avoid long OS-level connect hangs on Windows).
        try:
            parsed_base = urllib.parse.urlparse(self.cfg.base_url)
            host = parsed_base.hostname
            port = parsed_base.port or (443 if parsed_base.scheme == "https" else 80)
            if host:
                with socket.create_connection((host, port), timeout=min(2.0, float(self.cfg.timeout_s))):
                    pass
        except OSError as e:
            raise RuntimeError(
                "LLM server is not reachable. "
                f"Check LM Studio server at {self.cfg.base_url!r}. "
                f"(connect_error={e})"
            ) from e

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

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                last_err = None
                break
            except urllib.error.HTTPError as e:
                # Don't retry most HTTP errors (model/server will repeat).
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
                last_err = e
                if attempt < self.cfg.max_retries:
                    # Small backoff for transient network hiccups.
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "LLM request failed: "
                    f"{e}. Check that LM Studio server is reachable at {self.cfg.base_url!r} "
                    "and that the host/port are allowed by firewall."
                ) from e

        if last_err is not None:
            raise RuntimeError(f"LLM request failed: {last_err}") from last_err
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
    # LM Studio and other local OpenAI-compatible servers use http:// and ignore the key.
    if not api_key:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme == "http":
            api_key = "lm-studio"
        else:
            raise RuntimeError("OPENAI_API_KEY is not set.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "30").strip() or "30")
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "1").strip() or "1")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0").strip() or "0")
    # OpenAI-compatible API expects base like .../v1
    if base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return OpenAISarcasmClassifier(
        OpenAISarcasmConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            temperature=temperature,
        )
    )

