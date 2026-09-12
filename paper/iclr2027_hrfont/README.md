# HR-Font ICLR 2027 draft

This directory contains an anonymous, evidence-bounded first draft built from the official ICLR 2027 template.

## Build

```bash
python3 figures/gen_fig_method_overview.py
python3 figures/gen_fig_experiment_matrix.py
python3 figures/gen_fig_legacy_diagnostics.py
python3 figures/gen_fig_qualitative_template.py
latexmk -pdf main.tex
```

The workstation uses a Homebrew TeX Live installation. If `latexmk` is unavailable, the equivalent manual build is `pdflatex main.tex`, `bibtex main`, followed by two more `pdflatex main.tex` passes.

## Evidence rules

- Magenta `[TBD: ...]` text and `--` table cells are intentional placeholders.
- F0/F1/F2/F3 values are completed legacy diagnostics loaded from the repository; they are not results for Set-Delta or Graphics-Ref.
- Do not replace a placeholder until the corresponding run has exact code/data/checkpoint provenance.
- Do not turn E12 family compatibility into a universal design-quality claim.
- Keep the submission anonymous and leave `\iclrfinalcopy` disabled.

## Editable figures

Every figure has a Python source and PDF/SVG/PNG output. PDF and SVG are vector-editable in Illustrator, Inkscape, Affinity Designer, or Figma. The qualitative grid contains placeholders by design and must be populated with matched, preregistered examples.

## Primary PI review items

1. Confirm the one-sentence contribution and title.
2. Approve warp-only as the initial main path and keep value injection as D7.
3. Approve primitive-only keys plus learned Es12 values.
4. Use the fixed experiment seed 3407 throughout the paper.
5. Freeze validation-derived Go/No-Go thresholds before inspecting formal 40k results.
6. Confirm the minimum external set: FontDiffuser, FTransGAN, FCAGAN, and CF-Font; audit FSFont, DRA-font, and VQ-Font separately.
7. Complete font-license and human-evaluation ethics details.

The dependency-ordered run plan, stopping rules, compute accounting, and PI decisions are in [`EXPERIMENT_EXECUTION_PLAN.md`](EXPERIMENT_EXECUTION_PLAN.md).
