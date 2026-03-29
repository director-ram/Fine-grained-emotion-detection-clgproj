from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.predict import InferenceConfig, SarcasmPredictor


SMOKE_CASES = [
    ("It is raining outside.", "non-sarcastic"),
    ("Thank you for helping me.", "non-sarcastic"),
    ("Oh great, another production outage.", "sarcastic"),
    ("Yeah, because bugs are exactly what I wanted today.", "sarcastic"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test single-task sarcasm model.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=1,
        help="Minimum predictions required for each class in smoke test.",
    )
    args = parser.parse_args()

    predictor = SarcasmPredictor(InferenceConfig(model_dir=args.model_dir))
    predicted_labels: list[str] = []

    print("Running smoke set:")
    for text, expected in SMOKE_CASES:
        label, score = predictor.predict(text)
        predicted_labels.append(label)
        print(f"- text={text!r} expected={expected} predicted={label} score={score:.4f}")

    sarcastic_count = sum(1 for lbl in predicted_labels if lbl == "sarcastic")
    non_sarcastic_count = sum(1 for lbl in predicted_labels if lbl == "non-sarcastic")
    print(
        "Predicted counts: "
        f"non-sarcastic={non_sarcastic_count}, sarcastic={sarcastic_count}"
    )

    if sarcastic_count < args.min_class_count or non_sarcastic_count < args.min_class_count:
        raise SystemExit(
            "Smoke test failed: model predictions collapsed to one class. "
            "Retrain or recalibrate threshold before release."
        )
    print("Smoke test passed: both classes are present in outputs.")


if __name__ == "__main__":
    main()
