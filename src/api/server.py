from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predict import InferenceConfig, SarcasmPredictor


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sarcastic: bool
    label: str
    score: float


def create_app(model_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Sarcasm Detection API")

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
        predictor = SarcasmPredictor(InferenceConfig(model_dir=resolved_model_dir))

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        nonlocal predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        label, score = predictor.predict(req.text)
        sarcastic = label == "sarcastic"
        return PredictResponse(sarcastic=sarcastic, label=label, score=score)

    return app


app = create_app()

