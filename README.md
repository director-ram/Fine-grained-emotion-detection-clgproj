# Sarcasm Transformer Training

Backend-only project to train and evaluate a transformer-based model that detects sarcasm in individual sentences.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project layout

- `src/` – Python package with data, models, training, evaluation, and inference code.
- `configs/` – YAML configs for experiments.
- `data/` – Raw and processed datasets (not tracked).
- `scripts/` – Convenience scripts for running experiments.

## Usage (high level)

- Prepare datasets into a unified schema under `data/`.
- Configure experiment settings in `configs/*.yaml`.
- Run training via the training module.
- Use inference utilities or the optional API for predictions.

### 0. Download public tweet datasets (sarcasm + emotion)

To train the multi-task model on real tweets, first download and normalize the public datasets:

```bash
python scripts/download_public_datasets.py
```

This will create:

- `data/sarcasm/{train,val,test}.csv` with `text` and `sarcasm_label` (0 = non-irony, 1 = irony).
- `data/emotion/{train,val,test}.csv` with `text` and `emotion_label` (integer emotion ids).

### 1. Prepare a demo dataset

This project ships with a tiny demo sarcasm dataset for quick testing:

```bash
python scripts/prepare_demo_sarcasm_dataset.py
```

This will create stratified train/val/test splits under `data/processed/`.

### 2. Train a transformer model

Edit `configs/base.yaml` if needed, then run:

```bash
python -m src.training.train --config-path configs/base.yaml
```

The best model checkpoint will be saved under `outputs/<experiment_name>/best_model`.

### 2b. Train a multi-task sarcasm + emotion model on tweets

After running `scripts/download_public_datasets.py`, you can train the multi-task model with:

```bash
python -m src.training.train_multitask --config-path configs/multitask.yaml
```

This uses a shared BERT encoder with two heads (sarcasm + emotion) and saves the best multi-task checkpoint under:

- `outputs/multitask_tweets/best_model_multitask/`

### 3. Evaluate on the test set

```bash
python -m src.training.eval --model-dir outputs/<experiment_name>/best_model --test-path data/processed/test.csv
```

This prints metrics, a confusion matrix, and a classification report.

To evaluate the multi-task model on both sarcasm and emotion tweet test sets:

```bash
python -m src.training.eval_multitask --model-dir outputs/multitask_tweets/best_model_multitask
```

### 4. Run CLI inference

```bash
python -m src.inference.predict --model-dir outputs/<experiment_name>/best_model --text "Oh great, another bug in production."
```

For batch multi-task predictions (sarcasm + emotion) over a CSV of tweets:

```bash
python scripts/batch_predict_multitask.py ^
  --model-dir outputs/multitask_tweets/best_model_multitask ^
  --csv-path data/sarcasm/test.csv ^
  --text-col text
```

### 5. Start the FastAPI server

```bash
uvicorn src.api.server:app --reload
```

Then send a request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Yeah, because I love waiting in traffic.\"}"
```

The response includes `sarcastic` (boolean), `label`, and `score`.


