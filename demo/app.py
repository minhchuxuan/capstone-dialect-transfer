"""
Gradio demo — Vietnamese Bidirectional Dialect Transfer.

A single multi-task BARTpho-syllable model that performs:
  • Dialect → Standard normalization        (Feature A)
  • Standard → Dialect generation, by region (Feature C, novel)
  • Lexical normalization (teencode → std)   (Feature B)

The UI highlights the tokens the model changed, detects regional dialect
markers, estimates which region the output "sounds like", measures semantic
preservation, and compares against the rule-based baseline.

Usage:
    python -m demo.app                                   # loads results/checkpoints_v2/best
    python -m demo.app --model_path results/checkpoints_v2/best --share
    python -m demo.app --use_rules_only                  # no neural model (rules only)
"""
import argparse

import gradio as gr

from src.model.config import EvalConfig

# ---------------------------------------------------------------------------
# Globals (models are loaded once, lazily, and reused across requests)
# ---------------------------------------------------------------------------
_model = _tokenizer = _device = None
_sbert = None
MARKERS = EvalConfig().dialect_markers  # {northern, central, southern: [...]}

REGION_LABEL_TO_KEY = {"Northern": "northern", "Central": "central", "Southern": "southern"}
REGION_KEY_TO_LABEL = {v: k for k, v in REGION_LABEL_TO_KEY.items()}

MODE_D2S = "Dialect → Standard"
MODE_S2D = "Standard → Dialect"
MODE_LEX = "Lexical Normalization"


# ---------------------------------------------------------------------------
# Model loading / prediction
# ---------------------------------------------------------------------------

def load_model_if_needed(model_path: str):
    global _model, _tokenizer, _device
    if _model is not None:
        return
    from src.model.inference import load_model
    _model, _tokenizer, _device = load_model(model_path)


def model_predict(text: str, task: str) -> str:
    from src.model.inference import predict_single
    return predict_single(_model, _tokenizer, _device, text, task)


def rule_predict(text: str, task: str, region_key: str) -> str:
    from src.baselines.rule_baseline import predict
    return predict(text, task, region_key)


def get_sbert():
    """Lazy-load a multilingual sentence encoder for semantic-preservation scoring."""
    global _sbert
    if _sbert is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            _sbert = False
    return _sbert or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_task(mode: str, region_label: str) -> tuple[str, str]:
    if mode == MODE_D2S:
        return "dialect2std", "central"
    if mode == MODE_LEX:
        return "lexnorm", "central"
    rk = REGION_LABEL_TO_KEY.get(region_label, "central")
    return f"std2dialect_{rk}", rk


def _norm(tok: str) -> str:
    return tok.lower().strip(".,?!:;\"'()…")


def highlight_changes(src: str, out: str, mode: str) -> list[tuple[str, str | None]]:
    """Token-level highlight of what the model changed (added/substituted)."""
    src_set = {_norm(t) for t in src.split()}
    label = "dialect" if mode == MODE_S2D else "standard"
    spans = []
    for tok in out.split():
        spans.append((tok + " ", label if _norm(tok) and _norm(tok) not in src_set else None))
    return spans


def region_confidence(text: str) -> list[tuple[str, float]]:
    """Marker-based estimate of which region the text sounds like."""
    low = text.lower()
    counts = {r: sum(1 for m in MARKERS[r] if m in low) for r in MARKERS}
    total = sum(counts.values())
    if total == 0:
        return []
    return sorted(((REGION_KEY_TO_LABEL[r], c / total) for r, c in counts.items() if c),
                  key=lambda x: -x[1])


def semantic_similarity(a: str, b: str):
    m = get_sbert()
    if not m or not a.strip() or not b.strip():
        return None
    import numpy as np
    ea, eb = m.encode([a, b])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))


def build_analysis(task: str, region_key: str, src: str, out: str) -> str:
    lines = []
    if task.startswith("std2dialect"):
        region_label = REGION_KEY_TO_LABEL[region_key]
        found = [m for m in MARKERS[region_key] if m in out.lower()]
        lines.append(f"**Target region:** {region_label}")
        lines.append(f"**{region_label} markers produced:** "
                     + (", ".join(f"`{m}`" for m in found) if found else "_none_"))
        conf = region_confidence(out)
        if conf:
            lines.append("**Sounds like:** "
                         + " · ".join(f"{r} {p:.0%}" for r, p in conf))
    elif task == "dialect2std":
        removed = []
        for r, ms in MARKERS.items():
            removed += [m for m in ms if m in src.lower() and m not in out.lower()]
        lines.append("**Dialect markers normalized:** "
                     + (", ".join(f"`{m}`" for m in dict.fromkeys(removed)) if removed else "_none detected_"))
    else:
        lines.append("**Task:** lexical normalization (teencode → standard)")

    sim = semantic_similarity(src, out)
    if sim is not None:
        bar = "🟩" * round(sim * 10) + "⬜" * (10 - round(sim * 10))
        lines.append(f"**Meaning preserved:** {sim:.0%}  {bar}")
    changed = sum(1 for t in out.split() if _norm(t) not in {_norm(s) for s in src.split()})
    total = max(len(out.split()), 1)
    lines.append(f"**Tokens changed:** {changed}/{total} ({changed/total:.0%})")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main transfer callback
# ---------------------------------------------------------------------------

def transfer(text, mode, region_label, model_path, use_rules_only):
    if not text.strip():
        return "", [], "_Enter some text and press Transfer._", ""
    task, region_key = build_task(mode, region_label)

    if use_rules_only:
        model_out = "(neural model disabled — rules only)"
    else:
        try:
            load_model_if_needed(model_path)
            model_out = model_predict(text, task)
        except Exception as e:
            model_out = f"[model error: {e}]"

    spans = highlight_changes(text, model_out, mode) if not model_out.startswith("[") else [(model_out, None)]
    analysis = build_analysis(task, region_key, text, model_out)
    rule_out = rule_predict(text, task, region_key)
    if task == "lexnorm":
        rule_out = "(no rule baseline for lexical normalization)"
    return model_out, spans, analysis, rule_out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container {max-width: 1080px !important; margin: auto;}
#banner {background: linear-gradient(135deg,#0072B2 0%,#005a8e 100%); color:#fff;
  border-radius:14px; padding:20px 26px; margin-bottom:10px;}
#banner h1 {margin:0 0 4px 0; font-size:1.6rem;}
#banner p {margin:0; opacity:.92;}
.feat {display:inline-block; background:rgba(255,255,255,.16); border-radius:8px;
  padding:3px 10px; margin:6px 6px 0 0; font-size:.82rem;}
#out_box textarea {font-size:1.15rem !important; font-weight:600; color:#0a3d62;}
footer {visibility:hidden;}
"""

BANNER = """
<div id="banner">
  <h1>🗣️ Vietnamese Bidirectional Dialect Transfer</h1>
  <p>One multi-task <b>BARTpho-syllable</b> model · Northern · Central · Southern</p>
  <span class="feat">A · Dialect → Standard</span>
  <span class="feat">C · Standard → Dialect (region-conditioned)</span>
  <span class="feat">B · Teencode → Standard</span>
</div>
"""


def create_demo(model_path: str = "results/checkpoints_v2/best", use_rules_only: bool = False):
    color_map = {"dialect": "#E69F00", "standard": "#009E73"}

    with gr.Blocks(title="Vietnamese Dialect Transfer",
                   theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
                   css=CSS) as demo:
        gr.HTML(BANNER)

        with gr.Row():
            with gr.Column(scale=5):
                mode = gr.Radio([MODE_S2D, MODE_D2S, MODE_LEX], value=MODE_S2D, label="Mode")
                region = gr.Radio(["Northern", "Central", "Southern"], value="Central",
                                  label="Target region (Standard → Dialect)")
                input_text = gr.Textbox(label="Input", lines=3,
                                        placeholder="Nhập văn bản tiếng Việt…  e.g. Anh ấy đi đâu rồi?")
                with gr.Row():
                    submit_btn = gr.Button("Transfer  →", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

            with gr.Column(scale=5):
                model_output = gr.Textbox(label="✨ Model output", lines=2, elem_id="out_box")
                highlighted = gr.HighlightedText(label="What changed (orange = dialectal, green = standardized)",
                                                 color_map=color_map, combine_adjacent=True, show_legend=False)
                analysis = gr.Markdown("_Analysis will appear here._")
                with gr.Accordion("Compare with rule-based baseline", open=False):
                    rule_output = gr.Textbox(label="Rule-based baseline", lines=2)

        gr.Examples(
            examples=[
                ["Anh ấy đi đâu rồi?", MODE_S2D, "Central"],
                ["Mẹ tôi bảo trời sắp mưa rồi.", MODE_S2D, "Northern"],
                ["Cái gì mà gọi tôi hoài vậy bạn.", MODE_S2D, "Southern"],
                ["Tại sao bạn không trả lời tin nhắn của tôi?", MODE_S2D, "Central"],
                ["Ảnh đi mô rồi?", MODE_D2S, "Central"],
                ["Bả nói chi rứa?", MODE_D2S, "Central"],
                ["ko bít lm sao h", MODE_LEX, "Central"],
            ],
            inputs=[input_text, mode, region],
            label="Examples (click to load)",
        )

        with gr.Accordion("About this demo", open=False):
            gr.Markdown(
                "This is a single multi-task sequence-to-sequence model (BARTpho-syllable, "
                "≈396M params) selected by a text **task prefix**. **Standard → Dialect** "
                "generation is the novel contribution — it is *one-to-many* (a standard "
                "sentence has many valid dialectal forms), so the same input can be phrased "
                "differently per region. Northern is the hardest region (closest to standard, "
                "least training data). Outputs are model generations and may vary.")

        # Region selector only matters for Standard → Dialect
        def toggle_region(m):
            return gr.update(interactive=(m == MODE_S2D),
                             label="Target region (Standard → Dialect)" if m == MODE_S2D
                             else "Target region (n/a for this mode)")
        mode.change(toggle_region, inputs=mode, outputs=region)

        inputs = [input_text, mode, region]
        outs = [model_output, highlighted, analysis, rule_output]
        submit_btn.click(lambda t, m, r: transfer(t, m, r, model_path, use_rules_only),
                         inputs=inputs, outputs=outs)
        input_text.submit(lambda t, m, r: transfer(t, m, r, model_path, use_rules_only),
                          inputs=inputs, outputs=outs)
        clear_btn.click(lambda: ("", [], "_Analysis will appear here._", ""), outputs=outs)

    return demo


def main():
    p = argparse.ArgumentParser(description="Vietnamese dialect transfer demo")
    p.add_argument("--model_path", type=str, default="results/checkpoints_v2/best")
    p.add_argument("--use_rules_only", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    if not args.use_rules_only:
        print(f"Loading model from {args.model_path} …")
        load_model_if_needed(args.model_path)  # preload so the first request is instant
        print("Model ready.")

    demo = create_demo(args.model_path, args.use_rules_only)
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
