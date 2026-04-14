"""
Gradio demo: Vietnamese Bidirectional Dialect Transfer.

Features:
  - Toggle between Dialect→Standard and Standard→Dialect
  - Region selection for Standard→Dialect
  - Analysis tab: dialect markers, semantic similarity
  - Comparison tab: model vs baselines

Usage:
    python -m demo.app --model_path results/checkpoints/best
    # or for testing without model:
    python -m demo.app --use_rules_only
"""
import argparse
import json
from pathlib import Path

import gradio as gr

# Lazy imports for model (may not be available during development)
_model = None
_tokenizer = None
_device = None


def load_model_if_needed(model_path: str):
    """Lazily load the seq2seq model."""
    global _model, _tokenizer, _device
    if _model is not None:
        return
    from src.model.inference import load_model
    _model, _tokenizer, _device = load_model(model_path)


def model_predict(text: str, task: str) -> str:
    """Generate using the trained model."""
    if _model is None:
        return "[Model not loaded]"
    from src.model.inference import predict_single
    return predict_single(_model, _tokenizer, _device, text, task)


def rule_predict(text: str, task: str, region: str) -> str:
    """Generate using rule-based baseline."""
    from src.baselines.rule_baseline import predict
    return predict(text, task, region)


def dict_predict(text: str, task: str, region: str) -> str:
    """Placeholder for dictionary baseline (needs training data loaded)."""
    return f"[Dictionary baseline — needs index built from training data]"


# ---------------------------------------------------------------------------
# Core transfer functions
# ---------------------------------------------------------------------------

def transfer_text(
    text: str,
    mode: str,
    region: str,
    model_path: str = "",
    use_rules_only: bool = False,
) -> tuple[str, str, str]:
    """Main transfer function. Returns (model_output, rule_output, analysis)."""
    if not text.strip():
        return "", "", ""

    # Determine task
    if mode == "Dialect → Standard":
        task = "dialect2std"
    elif mode == "Standard → Dialect":
        region_key = {"Northern": "north", "Central": "central", "Southern": "south"}
        r = region_key.get(region, "central")
        task = f"std2dialect_{r}"
    else:  # Lexical Normalization
        task = "lexnorm"

    # Rule-based prediction
    region_key = {"Northern": "north", "Central": "central", "Southern": "south"}
    r = region_key.get(region, "central")
    rule_output = rule_predict(text, task, r)

    # Model prediction
    if use_rules_only:
        model_output = "[Model not loaded — showing rule-based only]"
    else:
        try:
            load_model_if_needed(model_path)
            model_output = model_predict(text, task)
        except Exception as e:
            model_output = f"[Model error: {e}]"

    # Analysis
    from src.model.config import EvalConfig
    markers = EvalConfig().dialect_markers
    analysis_lines = []

    if task.startswith("std2dialect"):
        region_markers = markers.get(r, [])
        found = [m for m in region_markers if m in model_output.lower() or m in rule_output.lower()]
        if found:
            analysis_lines.append(f"Dialect markers detected: {', '.join(found)}")
        else:
            analysis_lines.append("No dialect markers detected in output")
        analysis_lines.append(f"Target region: {region}")
    elif task == "dialect2std":
        # Check which dialect features were removed
        for reg, reg_markers in markers.items():
            found_in_src = [m for m in reg_markers if m in text.lower()]
            if found_in_src:
                analysis_lines.append(f"Input dialect markers ({reg}): {', '.join(found_in_src)}")

    analysis = "\n".join(analysis_lines) if analysis_lines else "No analysis available"
    return model_output, rule_output, analysis


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def create_demo(model_path: str = "", use_rules_only: bool = False):
    """Build the Gradio interface."""

    def on_submit(text, mode, region):
        return transfer_text(text, mode, region, model_path, use_rules_only)

    with gr.Blocks(
        title="Vietnamese Bidirectional Dialect Transfer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Vietnamese Bidirectional Dialect Transfer\n"
            "Transfer between standard Vietnamese and regional dialects "
            "(Northern, Central, Southern)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=["Dialect → Standard", "Standard → Dialect",
                             "Lexical Normalization"],
                    value="Standard → Dialect",
                    label="Mode",
                )
                region = gr.Radio(
                    choices=["Northern", "Central", "Southern"],
                    value="Central",
                    label="Target Region (for Std → Dialect)",
                )
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Nhập văn bản tiếng Việt...",
                    lines=3,
                )
                submit_btn = gr.Button("Transfer", variant="primary")

            with gr.Column(scale=1):
                model_output = gr.Textbox(label="Model Output", lines=3)
                rule_output = gr.Textbox(label="Rule-based Baseline", lines=3)
                analysis = gr.Textbox(label="Analysis", lines=4)

        # Examples
        gr.Examples(
            examples=[
                ["Anh ấy đi đâu rồi?", "Standard → Dialect", "Central"],
                ["Tại sao bạn không đến?", "Standard → Dialect", "Central"],
                ["Bà ấy nói gì vậy?", "Standard → Dialect", "Southern"],
                ["Ông ấy không biết gì cả", "Standard → Dialect", "Southern"],
                ["Ảnh đi mô rồi?", "Dialect → Standard", "Central"],
                ["Bả nói chi rứa?", "Dialect → Standard", "Central"],
                ["ko bít j hết á", "Lexical Normalization", "Central"],
            ],
            inputs=[input_text, mode, region],
        )

        submit_btn.click(
            on_submit,
            inputs=[input_text, mode, region],
            outputs=[model_output, rule_output, analysis],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo")
    parser.add_argument("--model_path", type=str, default="results/checkpoints/best")
    parser.add_argument("--use_rules_only", action="store_true",
                        help="Use only rule-based baseline (no model needed)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.model_path, args.use_rules_only)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
