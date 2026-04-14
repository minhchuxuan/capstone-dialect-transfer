"""
PhoBERT-based dialect region classifier.

Trains a simple classifier to predict which region (north/central/south)
a text belongs to. Used for:
  1. Region Accuracy metric: verify generated dialect matches target region.
  2. Auto-routing: detect if input is dialectal and which region.
  3. Downstream evaluation: measure if generated dialect is realistic.

Usage:
    # Train classifier
    python -m src.analysis.dialect_classifier --mode train

    # Predict region for a text
    python -m src.analysis.dialect_classifier --mode predict --text "Ảnh đi mô rồi?"

    # Evaluate model outputs
    python -m src.analysis.dialect_classifier --mode evaluate \
        --predictions results/metrics/model_predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset

from src.model.config import DataConfig, PROJECT_ROOT

cfg = DataConfig()

LABEL2ID = {"north": 0, "central": 1, "south": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
PHOBERT_NAME = "vinai/phobert-base"


def load_dialect_data(jsonl_path: Path, direction: str = "forward") -> HFDataset:
    """Load dialect text with region labels for classification."""
    texts = []
    labels = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["task"] == "dialect2std" and direction == "forward":
                region = r.get("region", "")
                if region in LABEL2ID:
                    texts.append(r["source"])  # dialect text
                    labels.append(LABEL2ID[region])
            elif r["task"].startswith("std2dialect") and direction == "reverse":
                region = r.get("region", "")
                if region in LABEL2ID:
                    texts.append(r["target"])  # dialect text (target of reverse)
                    labels.append(LABEL2ID[region])
    return HFDataset.from_dict({"text": texts, "label": labels})


def tokenize_fn(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"], max_length=max_length, truncation=True, padding=False,
    )


def train_classifier(output_dir: str = "results/dialect_classifier"):
    """Fine-tune PhoBERT for region classification."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PHOBERT_NAME, num_labels=NUM_LABELS,
    )

    # Load data — use dialect text (forward direction) for training
    train_ds = load_dialect_data(cfg.processed_dir / "train.jsonl", "forward")
    dev_ds = load_dialect_data(cfg.processed_dir / "dev.jsonl", "forward")
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    dev_ds = dev_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.argmax(-1)
        labels = eval_pred.label_ids
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_path / "best"))
    tokenizer.save_pretrained(str(output_path / "best"))
    print(f"Classifier saved to {output_path / 'best'}")


def predict_region(text: str, model_path: str = "results/dialect_classifier/best"):
    """Predict the dialect region of a text."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    return {
        "predicted_region": ID2LABEL[pred_id],
        "confidence": probs[pred_id].item(),
        "all_probs": {ID2LABEL[i]: probs[i].item() for i in range(NUM_LABELS)},
    }


def evaluate_region_accuracy(
    pred_path: Path,
    classifier_path: str = "results/dialect_classifier/best",
) -> dict:
    """Evaluate whether generated dialect text is classified to the correct region."""
    tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["task"].startswith("std2dialect") and r.get("region") in LABEL2ID:
                records.append(r)

    correct = 0
    total = 0
    for r in records:
        inputs = tokenizer(
            r["prediction"], return_tensors="pt", truncation=True, max_length=128,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = logits.argmax(dim=-1).item()
        if ID2LABEL[pred_id] == r["region"]:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {"region_accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Dialect region classifier")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--predictions", type=str, default="")
    parser.add_argument("--classifier_path", type=str, default="results/dialect_classifier/best")
    args = parser.parse_args()

    if args.mode == "train":
        train_classifier()
    elif args.mode == "predict":
        result = predict_region(args.text, args.classifier_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.mode == "evaluate":
        result = evaluate_region_accuracy(Path(args.predictions), args.classifier_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
