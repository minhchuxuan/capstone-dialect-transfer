"""
Back-translation pipeline for data augmentation.

Strategy (Edunov et al., 2018):
  1. Use a trained forward model (dialect→standard) to back-translate
     standard monolingual text into pseudo-dialect (using SAMPLING).
  2. Filter pseudo-pairs by quality (round-trip BLEU, length ratio).
  3. Write filtered pairs as augmented training data.

Usage:
    python -m src.data.augment_bt \
        --forward_model_path results/checkpoints/dialect2std_best \
        --monolingual_file data/monolingual/standard_vi.txt \
        --output_file data/augmented/bt_round1.jsonl \
        --round 1
"""
import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model.config import DataConfig, ModelConfig

cfg_data = DataConfig()
cfg_model = ModelConfig()


def load_monolingual(path: Path, max_lines: int = 100_000) -> list[str]:
    """Load standard Vietnamese monolingual sentences."""
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 10 and len(line.split()) <= 60:
                sentences.append(line)
                if len(sentences) >= max_lines:
                    break
    print(f"Loaded {len(sentences)} monolingual sentences from {path}")
    return sentences


def generate_pseudo_dialect(
    model,
    tokenizer,
    sentences: list[str],
    region: str,
    batch_size: int = 16,
    max_length: int = 128,
) -> list[dict]:
    """Generate pseudo-dialect using SAMPLING (not beam search) for diversity."""
    device = model.device
    results = []
    prefix = f"std2dialect_{region}: "

    for i in tqdm(range(0, len(sentences), batch_size), desc=f"BT [{region}]"):
        batch = sentences[i : i + batch_size]
        inputs = [prefix + s for s in batch]
        encoded = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_length,
                do_sample=True,              # SAMPLING — key for BT diversity
                temperature=cfg_model.bt_temperature,
                top_k=cfg_model.bt_top_k,
                top_p=cfg_model.bt_top_p,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for src, tgt in zip(batch, decoded):
            results.append({
                "task": f"std2dialect_{region}",
                "region": region,
                "source": src,
                "target": tgt,
                "meta": {
                    "dataset": "bt_synthetic",
                    "split": "train",
                    "direction": "reverse",
                    "bt_method": "sampling",
                },
            })
    return results


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def length_ratio_ok(src: str, tgt: str, threshold: float = 0.3) -> bool:
    """Reject if output length deviates too much from input."""
    len_src = max(len(src.split()), 1)
    len_tgt = len(tgt.split())
    ratio = abs(len_tgt / len_src - 1.0)
    return ratio <= threshold


def has_dialect_markers(text: str, region: str) -> bool:
    """Check if generated text contains at least 1 known dialect marker."""
    from src.model.config import EvalConfig
    markers = EvalConfig().dialect_markers.get(region, [])
    text_lower = text.lower()
    return any(m in text_lower for m in markers)


def round_trip_filter(
    pairs: list[dict],
    forward_model,
    forward_tokenizer,
    threshold: float = 0.5,
    batch_size: int = 16,
) -> list[dict]:
    """Keep pairs where round-trip (pseudo_dialect → standard) reconstructs
    the original standard sentence with BLEU >= threshold."""
    from sacrebleu.metrics import BLEU
    bleu_scorer = BLEU(effective_order=True)

    device = forward_model.device
    kept = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="Round-trip filter"):
        batch = pairs[i : i + batch_size]
        # Forward: pseudo-dialect → standard
        inputs = ["dialect2std: " + p["target"] for p in batch]
        encoded = forward_tokenizer(
            inputs, max_length=128, truncation=True, padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = forward_model.generate(
                **encoded, max_new_tokens=128, num_beams=4,
            )
        decoded = forward_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pair, reconstructed in zip(batch, decoded):
            original = pair["source"]
            score = bleu_scorer.sentence_score(reconstructed, [original]).score / 100.0
            if score >= threshold:
                kept.append(pair)

    return kept


def filter_pairs(pairs: list[dict]) -> list[dict]:
    """Apply non-model-based filters (length ratio, dialect markers)."""
    filtered = []
    for p in pairs:
        src, tgt, region = p["source"], p["target"], p["region"]
        if not length_ratio_ok(src, tgt, cfg_data.bt_length_ratio_max):
            continue
        if not has_dialect_markers(tgt, region):
            continue
        if src.strip() == tgt.strip():
            continue  # reject copy
        filtered.append(p)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Back-translation augmentation")
    parser.add_argument("--forward_model_path", type=str, required=True,
                        help="Path to trained dialect→standard model checkpoint")
    parser.add_argument("--monolingual_file", type=str,
                        default=str(cfg_data.monolingual_dir / "standard_vi.txt"))
    parser.add_argument("--output_file", type=str,
                        default=str(cfg_data.augmented_dir / "bt_round1.jsonl"))
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--max_sentences", type=int, default=50_000)
    parser.add_argument("--apply_round_trip", action="store_true",
                        help="Apply round-trip BLEU filter (slower but higher quality)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load forward model (for back-translation, we generate in reverse direction)
    print(f"Loading model from {args.forward_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.forward_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.forward_model_path).to(device)
    model.eval()

    # Load monolingual standard Vietnamese
    sentences = load_monolingual(Path(args.monolingual_file), args.max_sentences)

    # Generate for each region
    all_pairs = []
    for region in ("northern", "central", "southern"):
        pairs = generate_pseudo_dialect(model, tokenizer, sentences, region)
        pairs = filter_pairs(pairs)
        print(f"  {region}: {len(pairs)} pairs after basic filtering")
        all_pairs.extend(pairs)

    # Optional: round-trip filter
    if args.apply_round_trip:
        print("Applying round-trip BLEU filter...")
        all_pairs = round_trip_filter(
            all_pairs, model, tokenizer, cfg_data.bt_round_trip_bleu_threshold,
        )
        print(f"  After round-trip filter: {len(all_pairs)} pairs")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            p["meta"]["bt_round"] = args.round
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_pairs)} augmented pairs to {output_path}")


if __name__ == "__main__":
    main()
