"""
Retrieval baseline: find the nearest training example by sentence embedding
similarity and return its paired target.

Uses sentence-transformers for encoding. Works for both directions:
  - dialect→standard: find closest dialect in train, return its standard pair.
  - standard→dialect: find closest standard in train, return its dialect pair.
"""
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.model.config import DataConfig

cfg = DataConfig()

# Lazy-loaded globals
_model = None
_index = None  # list of (embedding, target_text)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # paraphrase-multilingual covers Vietnamese well
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def build_index(
    train_path: Path,
    tasks: list[str],
) -> list[dict]:
    """Build an in-memory index of (source_embedding, record) from training data."""
    model = _get_model()
    records = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["task"] in tasks:
                records.append(r)

    print(f"Encoding {len(records)} training examples for retrieval index...")
    sources = [r["source"] for r in records]
    embeddings = model.encode(sources, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    index = []
    for emb, rec in zip(embeddings, records):
        index.append({"embedding": emb, "record": rec})
    return index


def retrieve(query: str, index: list[dict], top_k: int = 1) -> list[dict]:
    """Find the nearest neighbor(s) by cosine similarity."""
    model = _get_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = []
    for item in index:
        sim = float(np.dot(q_emb, item["embedding"]))
        scores.append((sim, item["record"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scores[:top_k]]


def predict(source: str, index: list[dict]) -> str:
    """Return the target of the nearest training example."""
    neighbors = retrieve(source, index, top_k=1)
    if neighbors:
        return neighbors[0]["target"]
    return source


def run_on_file(
    test_path: Path,
    train_path: Path,
    output_path: Path,
    tasks: list[str],
):
    """Run retrieval baseline."""
    index = build_index(train_path, tasks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(test_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if json.loads(l)["task"] in tasks]

    for record in tqdm(lines, desc="Retrieval baseline"):
        results.append({
            "task": record["task"],
            "region": record.get("region"),
            "source": record["source"],
            "target": record["target"],
            "prediction": predict(record["source"], index),
            "baseline": "retrieval",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Retrieval baseline: {len(results)} predictions → {output_path}")
    return results


if __name__ == "__main__":
    train = cfg.processed_dir / "train.jsonl"
    test = cfg.processed_dir / "test.jsonl"
    out_dir = cfg.processed_dir.parent.parent / "results" / "metrics"

    # Forward: dialect→standard
    run_on_file(test, train, out_dir / "retrieval_fwd_predictions.jsonl",
                tasks=["dialect2std"])
    # Reverse: standard→dialect
    run_on_file(test, train, out_dir / "retrieval_rev_predictions.jsonl",
                tasks=["std2dialect_north", "std2dialect_central", "std2dialect_south"])
