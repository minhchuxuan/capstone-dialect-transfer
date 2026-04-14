"""
Multi-task seq2seq training for bidirectional Vietnamese dialect transfer.

Supports:
  - dialect2std (Feature A)
  - std2dialect_<region> (Feature C)
  - lexnorm (Feature B)
  - spell (optional)

Usage:
    python -m src.model.train
    python -m src.model.train --model_name vinai/bartpho-syllable --epochs 10
    python -m src.model.train --model_name VietAI/vit5-base --epochs 5
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset

from src.model.config import ModelConfig, DataConfig, PROJECT_ROOT

cfg_model = ModelConfig()
cfg_data = DataConfig()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_augmented(augmented_dir: Path) -> list[dict]:
    """Load all back-translation augmented files."""
    records = []
    if augmented_dir.exists():
        for f in sorted(augmented_dir.glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    records.append(json.loads(line))
    return records


def format_input(record: dict) -> str:
    """Prepend task prefix to source text."""
    task = record["task"]
    source = record["source"]
    # Task prefix is the task name itself, followed by ": "
    return f"{task}: {source}"


def prepare_dataset(records: list[dict]) -> HFDataset:
    """Convert records to HuggingFace Dataset with input/target strings."""
    inputs = []
    targets = []
    for r in records:
        inputs.append(format_input(r))
        targets.append(r["target"])
    return HFDataset.from_dict({"input_text": inputs, "target_text": targets})


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_fn(examples, tokenizer, max_source_length, max_target_length):
    """Tokenize inputs and targets for seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ---------------------------------------------------------------------------
# Metrics (computed during training for early stopping)
# ---------------------------------------------------------------------------

def compute_metrics_factory(tokenizer):
    """Return a compute_metrics function for Seq2SeqTrainer."""
    from sacrebleu.metrics import BLEU
    import numpy as np

    bleu_scorer = BLEU()

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Newer transformers versions can pass predictions as tuples or logits.
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # If logits are provided, convert to token ids.
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        # Replace ignore index before decoding.
        preds = np.where(preds == -100, pad_token_id, preds)
        labels = np.where(labels == -100, pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        result = bleu_scorer.corpus_score(decoded_preds, [decoded_labels])
        return {"bleu": result.score}

    return compute_metrics


def build_trainer(model, training_args, train_ds, dev_ds, tokenizer, data_collator):
    """Build a Seq2SeqTrainer compatible with transformers v4 and v5 APIs."""
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_factory(tokenizer),
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=cfg_model.early_stopping_patience
            ),
        ],
    }
    try:
        return Seq2SeqTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'tokenizer'" not in str(exc):
            raise

        # transformers>=5 replaced `tokenizer` with `processing_class`
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer
        return Seq2SeqTrainer(**trainer_kwargs)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train multi-task dialect transfer model")
    parser.add_argument("--model_name", type=str, default=cfg_model.model_name)
    parser.add_argument("--epochs", type=int, default=cfg_model.num_epochs)
    parser.add_argument("--batch_size", type=int, default=cfg_model.batch_size)
    parser.add_argument("--lr", type=float, default=cfg_model.learning_rate)
    parser.add_argument("--output_dir", type=str, default=cfg_model.output_dir)
    parser.add_argument("--include_augmented", action="store_true",
                        help="Include back-translation augmented data")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Train only on specific tasks (e.g., dialect2std lexnorm)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading training data...")
    train_records = load_jsonl(cfg_data.processed_dir / "train.jsonl")
    dev_records = load_jsonl(cfg_data.processed_dir / "dev.jsonl")

    if args.include_augmented:
        aug = load_augmented(cfg_data.augmented_dir)
        print(f"  Adding {len(aug)} augmented records")
        train_records.extend(aug)

    # Filter by tasks if specified
    if args.tasks:
        train_records = [r for r in train_records if r["task"] in args.tasks]
        dev_records = [r for r in dev_records if r["task"] in args.tasks]

    print(f"  Train: {len(train_records)} records")
    print(f"  Dev:   {len(dev_records)} records")

    # Task distribution
    task_counts: dict[str, int] = {}
    for r in train_records:
        task_counts[r["task"]] = task_counts.get(r["task"], 0) + 1
    print(f"  Task distribution: {task_counts}")

    # Prepare datasets
    train_ds = prepare_dataset(train_records)
    dev_ds = prepare_dataset(dev_records)

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Tokenize
    max_src = cfg_model.max_source_length
    max_tgt = cfg_model.max_target_length
    train_ds = train_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_src, max_tgt),
        batched=True,
        remove_columns=["input_text", "target_text"],
    )
    dev_ds = dev_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_src, max_tgt),
        batched=True,
        remove_columns=["input_text", "target_text"],
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=cfg_model.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=cfg_model.warmup_ratio,
        weight_decay=cfg_model.weight_decay,
        fp16=cfg_model.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=max_tgt,
        generation_num_beams=cfg_model.num_beams,
        logging_steps=50,
        save_total_limit=3,
        report_to="none",  # set to "wandb" if you have it configured
    )

    # Trainer
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_ds=train_ds,
        dev_ds=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save best model
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")

    # Evaluate on dev
    results = trainer.evaluate()
    print(f"Dev results: {results}")

    # Save results
    with open(output_dir / "dev_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
