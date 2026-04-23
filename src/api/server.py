from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Repo root: .../src/api/server.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("uvicorn.error")

from src.inference.predict import InferenceConfig, SarcasmPredictor
from src.inference.predict_multitask_sarcasm import (
    MultiTaskSarcasmInferenceConfig,
    MultiTaskSarcasmPredictor,
)
from src.llm.openai_sarcasm import load_openai_classifier_from_env
from src.llm.gemini_sarcasm import load_gemini_classifier_from_env
from src.safety.taboo_filter import contains_taboo

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sarcastic: bool
    label: str
    score: float
    message: str
    # "llm" = OpenAI-compatible API (LM Studio / OpenAI); scores are 0.0 or 1.0.
    # "local" = transformer checkpoint; score is calibrated probability (often between 0 and 1).
    source: str = "unknown"


def _optional_local_threshold() -> Optional[float]:
    """Raises precision (fewer false sarcastic) when SARCASM_BACKEND=local. Ignored for LM Studio."""
    raw = os.environ.get("SARCASM_LOCAL_THRESHOLD", "").strip()
    if not raw:
        return None
    return float(raw)


def _load_local_predictor(model_dir: Optional[Path]) -> object:
    # Prefer multi-task tweet checkpoint if present, because the demo single-task model
    # is too small and often collapses to all-sarcastic.
    threshold = _optional_local_threshold()
    resolved_model_dir = model_dir
    if resolved_model_dir is None:
        multitask_dir = Path("outputs") / "multitask_tweets" / "best_model_multitask"
        single_dir = Path("outputs") / "sarcasm_transformer_base" / "best_model"
        resolved_model_dir = multitask_dir if multitask_dir.exists() else single_dir

    if not resolved_model_dir.exists():
        raise RuntimeError(
            f"Model directory {resolved_model_dir} does not exist. "
            "Train a model first or point the API to an existing checkpoint."
        )

    if (resolved_model_dir / "multitask_config.json").exists():
        return MultiTaskSarcasmPredictor(
            MultiTaskSarcasmInferenceConfig(model_dir=resolved_model_dir, threshold=threshold)
        )
    return SarcasmPredictor(InferenceConfig(model_dir=resolved_model_dir, threshold=threshold))


class _FallbackPredictor:
    def __init__(self, primary: object, fallback: object) -> None:
        self.primary = primary
        self.fallback = fallback

    def predict(self, text: str):
        try:
            return self.primary.predict(text)  # type: ignore[attr-defined]
        except Exception:
            return self.fallback.predict(text)  # type: ignore[attr-defined]


def create_app(model_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Sarcasm Detection API")

    # Allow browser-based clients (e.g., Vite dev server) to call this API.
    # You can extend `allow_origins` for production by setting a stricter list.
    app.add_middleware(
        CORSMiddleware,
        # Dev-friendly: allow any localhost port (Vite may auto-pick 5174, 5175, ...)
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Cache the predictor (LLM, single-task, or multi-task sarcasm head).
    predictor: Optional[object] = None

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal predictor
        # Load .env from repo root (cwd-independent; fixes silent fallback to local BERT).
        if load_dotenv is not None:
            env_path = _REPO_ROOT / ".env"
            load_dotenv(env_path, override=True)
        backend = (os.environ.get("SARCASM_BACKEND", "local").strip().lower())
        logger.info(
            "SARCASM_BACKEND=%r dotenv=%s",
            backend,
            str(_REPO_ROOT / ".env"),
        )
        if backend in {"llm", "openai", "lmstudio"}:
            # OpenAI-compatible API only (LM Studio, OpenAI, etc.). Local BERT is off unless
            # SARCASM_LLM_FALLBACK=1 — avoids false positives from the small checkpoint.
            primary = load_openai_classifier_from_env()
            fallback_on = os.environ.get("SARCASM_LLM_FALLBACK", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if fallback_on:
                try:
                    fallback = _load_local_predictor(model_dir=model_dir)
                    predictor = _FallbackPredictor(primary=primary, fallback=fallback)
                    logger.warning(
                        "SARCASM_LLM_FALLBACK is on: local BERT may be used if the LLM errors."
                    )
                except Exception:
                    predictor = primary
            else:
                predictor = primary
            if not fallback_on:
                logger.info(
                    "Sarcasm: OpenAI-compatible LLM only (no local BERT; set SARCASM_LLM_FALLBACK=1 to allow fallback)."
                )
            return
        if backend == "gemini":
            predictor = load_gemini_classifier_from_env()
            return

        predictor = _load_local_predictor(model_dir=model_dir)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        nonlocal predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        if contains_taboo(req.text):
            return PredictResponse(
                sarcastic=False,
                label="non-sarcastic",
                score=0.0,
                message="these words are prohibited, please don't use them",
                source="safety",
            )

        try:
            label, score = predictor.predict(req.text)  # type: ignore[attr-defined]
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        sarcastic = label == "sarcastic"
        message = "yes, its sarcastic" if sarcastic else "no, its not sarcastic"
        # LLM adapters return exactly 0.0 or 1.0; local models return softmax probability.
        src = (
            "llm"
            if (score == 0.0 or score == 1.0)
            else "local"
        )
        return PredictResponse(
            sarcastic=sarcastic,
            label=label,
            score=score,
            message=message,
            source=src,
        )

    return app


app = create_app()

