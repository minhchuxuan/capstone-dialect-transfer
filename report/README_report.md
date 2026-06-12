# Report & slides — build instructions

LaTeX source for the capstone *"Bidirectional Vietnamese Dialect Transfer"* (IT4772E NLP, HUST 2025-2).

## Compile

The source uses **`fontspec`** for native Vietnamese rendering, so it needs a **Unicode TeX engine** (XeLaTeX, LuaLaTeX, or [Tectonic](https://tectonic-typesetting.github.io/)). The easiest, dependency-free option here is Tectonic (auto-downloads packages):

```bash
# report (resolves references + bibliography automatically)
tectonic main.tex
# slides
tectonic slides.tex
```

Or with a TeX Live install:

```bash
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
xelatex slides.tex && xelatex slides.tex
```

> Do **not** use `pdflatex` — it cannot render the Vietnamese precomposed glyphs via `fontspec`.

## Figures

Both documents `\includegraphics` from `../results/figures/` (via the `\figdir` macro) and expect these PDF figures, produced by `python -m src.analysis.make_figures`:
`region_imbalance.pdf`, `dataset_composition.pdf`, `baseline_comparison.pdf`, `main_results.pdf`, `copy_vs_model_bleu.pdf`, `dfr_by_region.pdf`.

## Files
- `main.tex` — full report.
- `slides.tex` — Beamer presentation.
- `references.bib` — bibliography.
