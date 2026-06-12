"""
GPU probes for the report:
  1. Tokenization comparison: how BARTpho's SentencePiece splits standard vs dialect
     tokens (illustrates syllable-friendly subword handling of dialect OOV).
  2. One-to-many diversity: sample K dialect realisations per standard sentence and
     measure Self-BLEU (lower = more diverse) — quantifies that sampling yields
     varied valid outputs, the core of the one-to-many argument.
  3. Qualitative old-vs-new comparison on fresh, out-of-corpus sentences.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.analysis.model_probe \
        --new results/checkpoints_v2/best --old results/checkpoints/best
"""
import argparse
import json
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FRESH = [
    "Hôm nay trời đẹp quá, chúng ta đi chơi đi.",
    "Tại sao bạn không trả lời tin nhắn của tôi?",
    "Mẹ tôi nấu ăn rất ngon.",
    "Anh ấy nói cái gì vậy, tôi không hiểu.",
    "Cô ấy đi đâu mà giờ vẫn chưa về?",
    "Bạn có biết bây giờ là mấy giờ không?",
    "Con không thích ăn cơm với cá đâu.",
    "Ông ấy bảo là mai sẽ đến nhà chơi.",
]
REGIONS = ["central", "southern", "northern"]


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSeq2SeqLM.from_pretrained(path).to("cuda").eval()
    return tok, m


def gen(tok, m, task, src, **kw):
    enc = tok(f"{task}: {src}", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=64, **kw)
    return tok.decode(out[0], skip_special_tokens=True)


def self_bleu(hyps):
    """Mean pairwise BLEU among a set of hypotheses (lower = more diverse)."""
    if len(hyps) < 2:
        return 0.0
    scores = []
    for i, h in enumerate(hyps):
        refs = [hyps[j] for j in range(len(hyps)) if j != i]
        scores.append(BLEU(effective_order=True).sentence_score(h, refs).score)
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="results/checkpoints_v2/best")
    ap.add_argument("--old", default="results/checkpoints/best")
    ap.add_argument("--out", default="results/analysis_probe.json")
    args = ap.parse_args()

    tok, m = load(args.new)
    result = {}

    # 1. Tokenization comparison
    pairs = [("Anh ấy đi đâu rồi?", "Ảnh đi mô rồi?"),
             ("Bà ấy nói gì vậy?", "Bả nói chi rứa?"),
             ("Không biết", "Hông biết")]
    toks_demo = []
    for std, dia in pairs:
        toks_demo.append({
            "standard": std, "standard_pieces": tok.tokenize(std),
            "dialect": dia, "dialect_pieces": tok.tokenize(dia),
        })
    result["tokenization"] = toks_demo

    # 2. One-to-many diversity (sampling) vs beam (should be ~no diversity)
    torch.manual_seed(0)
    div = {}
    for region in REGIONS:
        sb_sample, sb_beam = [], []
        examples = []
        for s in FRESH[:5]:
            samp = [gen(tok, m, f"std2dialect_{region}", s, do_sample=True,
                        top_p=0.95, temperature=1.1, num_beams=1) for _ in range(5)]
            sb_sample.append(self_bleu(samp))
            examples.append({"std": s, "samples": samp})
        div[region] = {"self_bleu_sampling": sum(sb_sample) / len(sb_sample),
                       "examples": examples}
    result["diversity"] = div

    # 3. Old vs new qualitative comparison
    tok_o, m_o = load(args.old)
    comp = []
    for s in FRESH:
        row = {"standard": s}
        for region in REGIONS:
            row[f"{region}_old"] = gen(tok_o, m_o, f"std2dialect_{region}", s, num_beams=4)
            row[f"{region}_new"] = gen(tok, m, f"std2dialect_{region}", s, num_beams=4)
        comp.append(row)
    result["old_vs_new"] = comp

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")
    print("\n=== Self-BLEU (sampling; lower=more diverse) ===")
    for region in REGIONS:
        print(f"  {region}: {div[region]['self_bleu_sampling']:.1f}")
    print("\n=== Old vs New (central) ===")
    for row in comp[:6]:
        print(f"  STD: {row['standard']}")
        print(f"    old: {row['central_old']}")
        print(f"    new: {row['central_new']}")


if __name__ == "__main__":
    main()
