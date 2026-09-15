# HR-Font: Completing Font Families across Scripts

2026-09-15 I-series revision. Main document: [main.tex](main.tex).
Review PDF: [hrfont_iclr2027_draft.pdf](output/pdf/hrfont_iclr2027_draft.pdf).

The draft presents three contributions: cross-script completion and evaluation, Dynamic Delta, and online spatial target-character appearance completion (TC).
It follows the implemented I1 architecture. I2 is an auxiliary-supervision study awaiting rendering approval, not a fourth contribution. Results remain blank until matched generation and evaluator validation are finalized.

## Collaborator entry points

- [Changes and rationale](REVISION_NOTES_20260915_I.md)
- [Authoritative 8-V100 plan](../../reports/I_EXECUTION_20260915.md)
- [Remaining paper work](RESULTS_TODO.md)
- [Current review checklist](PAPER_ISSUE_TRACKER.md)
- [Paper experimental plan](EXPERIMENT_EXECUTION_PLAN.md)
- [I2 rendering audit](../../reports/review_20260915/I2_RENDER_REVIEW.md)
- [Latest E12 audit](../../reports/review_20260915/E12_REVIEW.md)

## Build

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    cp main.pdf output/pdf/hrfont_iclr2027_draft.pdf

Use pdfLaTeX and set the Overleaf main document to main.tex. The conference style files are preserved.
Figures 1 and 2 are editable LaTeX sources in figures/method_overview.tex and figures/qualitative_layout.tex. No image-generation service or external plot dependency is required for this revision.

Older Python-generated Set-Delta / Graphics-Ref diagrams and previous review notes remain historical assets. The active diagram reflects the current I implementation, not those earlier proposals. Earlier planning versions remain in Git history.

## Results and review

Empty table cells denote missing measurements. The abstract and conclusion presently describe the method; their empirical summary is added after the results are available.
E12-b is trained, but internal retrieval and human-correlation results require the documented review before final use. The initial human pilot is not empty and does not establish positive visual agreement.
The current I primary episode is 永和书风骨韵天地, exactly as implemented; the final4096 panel is validation. Other reference sets retain separate provenance.
The paper is a research draft, not a completed submission package. Font permissions, human evaluation, external baselines, and final submission requirements remain to be completed.
