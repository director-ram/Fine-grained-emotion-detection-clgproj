from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predict import InferenceConfig, SarcasmPredictor
from src.inference.predict_multitask_sarcasm import (
    MultiTaskSarcasmInferenceConfig,
    MultiTaskSarcasmPredictor,
)
from src.llm.openai_sarcasm import load_openai_classifier_from_env
from src.llm.gemini_sarcasm import load_gemini_classifier_from_env

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
        # Load .env for local dev (API keys, backend selection).
        if load_dotenv is not None:
            load_dotenv()
        backend = (os.environ.get("SARCASM_BACKEND", "local").strip().lower())
        if backend in {"llm", "openai", "lmstudio"}:
            # "lmstudio" uses the same OpenAI-compatible protocol, but typically points
            # OPENAI_BASE_URL to a local server like http://localhost:1234/v1
            predictor = load_openai_classifier_from_env()
            return
        if backend == "gemini":
            predictor = load_gemini_classifier_from_env()
            return

        # Prefer multi-task tweet checkpoint if present, because the demo single-task model
        # is too small and often collapses to all-sarcastic.
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
            predictor = MultiTaskSarcasmPredictor(
                MultiTaskSarcasmInferenceConfig(model_dir=resolved_model_dir)
            )
        else:
            predictor = SarcasmPredictor(InferenceConfig(model_dir=resolved_model_dir))

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        nonlocal predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        try:
            label, score = predictor.predict(req.text)  # type: ignore[attr-defined]
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        sarcastic = label == "sarcastic"
        message = "yes, its sarcastic" if sarcastic else "no, its not sarcastic"
        return PredictResponse(
            sarcastic=sarcastic,
            label=label,
            score=score,
            message=message,
        )

    return app


app = create_app()

