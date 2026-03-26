from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predict import InferenceConfig, SarcasmPredictor


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

    # Use a simple attribute on app state to cache the predictor
    predictor: Optional[SarcasmPredictor] = None

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal predictor
        resolved_model_dir = model_dir or Path("outputs") / "sarcasm_transformer_base" / "best_model"
        if not resolved_model_dir.exists():
            raise RuntimeError(
                f"Model directory {resolved_model_dir} does not exist. "
                "Train a model first or point the API to an existing checkpoint."
            )
        # Keep API inference fast for interactive usage (browser UI).
        predictor = SarcasmPredictor(
            InferenceConfig(model_dir=resolved_model_dir, max_seq_length=64)
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        nonlocal predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        label, score = predictor.predict(req.text)
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

