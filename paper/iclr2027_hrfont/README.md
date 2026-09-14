# HR-Font: Completing Font Families across Scripts

2026-09-14 revision. Main document: [main.tex](main.tex).
Review PDF: [hrfont_iclr2027_draft.pdf](output/pdf/hrfont_iclr2027_draft.pdf).

The draft presents three contributions: cross-script completion and evaluation, Mean-Delta, and target-character appearance completion (TC-v2).
It follows the implemented architecture. Quantitative results remain blank while matched evaluation and the learned evaluator are completed.

## Collaborator entry points

- [Changes and rationale](REVISION_NOTES_20260914.md)
- [Authoritative 8-V100 plan](../../reports/G_NEXT_8V100_PLAN_20260914.md)
- [Remaining paper work](RESULTS_TODO.md)
- [Current review checklist](PAPER_ISSUE_TRACKER.md)
- [TC implementation design](../../reports/G_STYLE_COMPLETION_PLAN_20260914.md)

## Build

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    cp main.pdf output/pdf/hrfont_iclr2027_draft.pdf

Use pdfLaTeX and set the Overleaf main document to main.tex. The conference style files are preserved.
Figures 1 and 2 are editable LaTeX sources in figures/method_overview.tex and figures/qualitative_layout.tex. No image-generation service or external plot dependency is required for this revision.

The older Python-generated Set-Delta / Graphics-Ref diagrams and the September 12 review notes remain as historical assets. They are not included in the active draft and must not be regenerated into the current paper. Earlier versions of the overwritten planning files are recoverable from Git history.

## Results and review

Empty table cells denote missing measurements. The abstract and conclusion presently describe the method; their empirical summary is added after the results are available.
The draft does not treat old E12 scores as the result of the currently training evaluator.
The primary episode is the confirmed new Ref8; the ongoing old-Ref8 board is tracked as a separate exploratory batch.
The paper is a research draft, not a completed submission package. Font permissions, human evaluation, external baselines, and final submission requirements remain to be completed.
