## Sarcasm Transformer Project Report

### Overview

This project fine-tunes a pretrained transformer model (default: `bert-base-uncased`) to classify individual sentences as sarcastic or non-sarcastic.

### Data

- Demo dataset of short English sentences labeled as sarcastic (1) or non-sarcastic (0).
- Data is normalized into a canonical schema with `text` and `label` columns and split into stratified train/val/test sets.

### Model & training

- Architecture: transformer encoder + classification head (`AutoModelForSequenceClassification`).
- Optimization: AdamW with linear warmup and weight decay.
- Training loop implemented via Hugging Face `Trainer` with early selection of the best model based on F1.

### Evaluation

- Metrics: accuracy, precision, recall, F1.
- Additional outputs: confusion matrix and full classification report on the test set.

### Inference

- CLI tool for single-sentence predictions.
- Optional FastAPI service with a `/predict` endpoint returning a sarcasm flag and confidence score.

