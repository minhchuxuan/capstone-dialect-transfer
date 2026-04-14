"""
Inference module: load a trained model and generate predictions.

Supports both beam search (for evaluation) and sampling (for back-translation).

Usage:
    python -m src.model.inference \
        --model_path results/checkpoints/best \
        --input_file data/processed/test.jsonl \
        --output_file results/metrics/model_predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model.config import ModelConfig, DataConfig, PROJECT_ROOT

cfg_model = ModelConfig()
cfg_data = DataConfig()


def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer from checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def format_input(record: dict) -> str:
    """Prepend task prefix to source text."""
    return f"{record['task']}: {record['source']}"


def generate_batch(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[str]:
    """Generate outputs for a batch of input texts."""
    encoded = tokenizer(
        texts,
        max_length=cfg_model.max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": 1,
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": num_beams,
        })

    with torch.no_grad():
        outputs = model.generate(**encoded, **gen_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def predict_file(
    model,
    tokenizer,
    device: str,
    input_path: Path,
    output_path: Path,
    tasks: list[str] | None = None,
    batch_size: int = 16,
    num_beams: int = 4,
):
    """Run inference on a JSONL file, write predictions."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if tasks and r["task"] not in tasks:
                continue
            records.append(r)

    print(f"Predicting {len(records)} examples (batch_size={batch_size})...")
    results = []
    for i in tqdm(range(0, len(records), batch_size)):
        batch = records[i : i + batch_size]
        inputs = [format_input(r) for r in batch]
        outputs = generate_batch(
            model, tokenizer, inputs, device,
            num_beams=num_beams,
            max_new_tokens=cfg_model.max_new_tokens,
        )
        for r, pred in zip(batch, outputs):
            results.append({
                "task": r["task"],
                "region": r.get("region"),
                "source": r["source"],
                "target": r["target"],
                "prediction": pred.strip(),
                "model": "seq2seq",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} predictions → {output_path}")
    return results


def predict_single(
    model,
    tokenizer,
    device: str,
    text: str,
    task: str,
    num_beams: int = 4,
) -> str:
    """Predict a single input. Convenience for demo/interactive use."""
    input_text = f"{task}: {text}"
    outputs = generate_batch(
        model, tokenizer, [input_text], device,
        num_beams=num_beams,
    )
    return outputs[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str,
                        default=str(cfg_data.processed_dir / "test.jsonl"))
    parser.add_argument("--output_file", type=str,
                        default=str(PROJECT_ROOT / "results" / "metrics" / "model_predictions.jsonl"))
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=cfg_model.num_beams)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)
    predict_file(
        model, tokenizer, device,
        Path(args.input_file),
        Path(args.output_file),
        tasks=args.tasks,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()
