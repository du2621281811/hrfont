# HR-Font ICLR 2027 paper issue tracker

Living list of paper problems for collaborators. Update this file when an item is opened, closed, or deferred. Do not treat a closed writing item as a license to fill magenta `[TBD]` cells.

| Field | Value |
|---|---|
| Paper commit audited | `71500abd` (`paper: sharpen cross-script story and experiment contract`) |
| Audit date | 2026-09-12 |
| Official template | [ICLR 2027 style files](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip); local copy used for the audit is the zip named `iclr-2027-style-files.zip` |
| Main source | [`main.tex`](main.tex) |
| Draft PDF in repo | [`output/pdf/hrfont_iclr2027_draft.pdf`](output/pdf/hrfont_iclr2027_draft.pdf) |
| Current verdict | Official template **compliant**. Not submission-ready: placeholders, empty result tables, two layout risks. |

Related files (do not duplicate their ledgers here):

- Writing review ops: [`REVIEW_OPERATIONAL_20260912.md`](REVIEW_OPERATIONAL_20260912.md)
- What `71500abd` already applied: [`REVISION_NOTES_20260912.md`](REVISION_NOTES_20260912.md)
- When a `[TBD]` / `--` cell may be filled: [`RESULTS_TODO.md`](RESULTS_TODO.md)
- Run order and Go/No-Go: [`EXPERIMENT_EXECUTION_PLAN.md`](EXPERIMENT_EXECUTION_PLAN.md)

## Status at a glance

| Area | Status | Notes |
|---|---|---|
| Official `iclr2027_conference.sty` / `.bst` | Pass | Byte-identical to the ICLR zip; sty was not forked |
| Anonymity | Pass | `\author{Anonymous authors}`; `\iclrfinalcopy` stays commented |
| Required AI statement; ethics; reproducibility | Pass | Unnumbered `subsection*` before `\bibliography` |
| Structure | Pass | Bibliography uses `iclr2027_conference`; `\appendix` after references |
| Main-text page limit (≤ 9 at submission) | Pass | Conclusion on PDF page 7; full PDF 10 pages with statements + refs + appendix |
| Layout polish | Open | Table 2 `\resizebox`; Figure 4 floats after the conclusion |
| Evidence / results | Blocked | Magenta TBD in abstract, results, conclusion; Table 2–3 still `--` |
| Overleaf | Documented | Upload only the compile set below; Main document = `main.tex` |

## How to manage this list

1. One row per issue. IDs are stable (`T-` template/layout, `C-` content/evidence, `O-` Overleaf/process, `W-` writing already handled).
2. Change **Status** to `open` / `doing` / `done` / `deferred`. Add a dated note in the changelog when you close an item.
3. Do not fill a `[TBD]` or `--` until [`RESULTS_TODO.md`](RESULTS_TODO.md) release conditions are met.
4. Keep the submission anonymous. Do not uncomment `\iclrfinalcopy` for the review PDF.
5. Prefer a paper-only commit when you change `main.tex`, `references.bib`, figures, or this tracker.

---

## Open issues

### T — template and layout

| ID | Severity | Status | Issue | Where | Action |
|---|---|---|---|---|---|
| T1 | Medium | open | Table 2 is wrapped in `\resizebox{\linewidth}{!}` | `main.tex` Table `tab:main` | Rebuild the table so it fits at `\small` without scaling below 10 pt |
| T2 | Medium | open | Figure 4 floats onto PDF page 8, after the conclusion, splitting the AI-use statement | `fig:qualitative` | Keep the figure in the experiment section (`[t]` / shorten pages 6–7 / `\clearpage` before statements) |
| T3 | Low | open | Folder still contains official example `iclr2027_conference.tex` and `iclr2027_conference.bib` | this directory | Do not compile them. Overleaf Main document must be `main.tex`. Optional: drop them from the Overleaf zip only; keep in git as the official shell |
| T4 | Info | done | Style files vs official zip | `.sty` `.bst` `math_commands.tex` `fancyhdr.sty` `natbib.sty` | Audited 2026-09-12: identical to the ICLR 2027 zip |

### C — content and evidence (blocks camera-ready / full-paper PDF)

Magenta `\tbd{...}` in **main text** (6):

| ID | Status | Location | Placeholder |
|---|---|---|---|
| C1 | open | Abstract | dataset and evaluation protocol; headline family/style result; identity result; Set-Delta and Graphics-Ref findings |
| C2 | open | After Table 2 | headline preference / identity / geometry sentence |
| C3 | open | Conclusion | summarize headline result and strongest mechanism finding |

Magenta `\tbd{...}` in **appendix** (do not count toward the 9-page limit):

| ID | Status | Location | Placeholder |
|---|---|---|---|
| C4 | open | Appendix A | parent checkpoint SHA; code revision; data hashes; optimizer; hardware; runtime |
| C5 | open | Appendix C | human-eval sample size / power analysis |
| C6 | open | Appendix D | n-shot curve; script/stratum tables; leakage probes; efficiency; failure-case grid |

Empty `--` tables that still need protocol-gated numbers:

| ID | Status | Table / figure | Release gate |
|---|---|---|---|
| C7 | open | Table 2 main comparison | frozen test opened once; clustered CIs |
| C8 | open | Table 3 mechanism attribution | each retained claim has its matched control |
| C9 | open | Figure 4 qualitative grid | preregistered row-selection rule; matched noise/sampler/refs |

### O — Overleaf and collaborator compile

| ID | Severity | Status | Issue | Action |
|---|---|---|---|---|
| O1 | High | open | Compiling the official example file produces the ICLR *formatting instructions* paper, not HR-Font | Set Overleaf Main document to `main.tex`; compiler **pdfLaTeX**. Never upload the ICLR zip as the whole project |
| O2 | Info | done | Compile file set for Overleaf | See [Overleaf pack](#overleaf-pack) below |

### W — writing review (applied in `71500abd`, do not re-open unless regression)

Tracked originally in [`REVIEW_OPERATIONAL_20260912.md`](REVIEW_OPERATIONAL_20260912.md); decisions in [`REVISION_NOTES_20260912.md`](REVISION_NOTES_20260912.md).

| ID | Status | Residual |
|---|---|---|
| W-A1 | done (modified) | D2−D1 is a geometry **package** comparison. D1n stays **conditional** |
| W-A2 | done | FTransGAN, FCAGAN, DRA-font cited with verified DOI. DRA-font stays related work until a faithful executable protocol exists |
| W-A3 | done | Self-questioning sentence removed from Method |
| W-A4 | done | Baseline protocol cards added |
| W-A5 | done | Residual vs fused absolute features stated in related work |
| W-A6 | done (modified) | Alpha kept as retrieval/logit prior; entropy diagnostics before any routing claim |
| W-D1–D8 | done (selective) | Defensive because-clauses removed or rewritten as protocol facts |

---

## Overleaf pack

Do **not** upload `iclr-2027-style-files.zip` as the project. That zip’s main tex is the official instructions paper.

Upload a project whose root contains:

```
main.tex
references.bib
math_commands.tex
iclr2027_conference.sty
iclr2027_conference.bst
fancyhdr.sty
natbib.sty
figures/fig_method_overview.pdf
figures/fig_experiment_matrix.pdf
figures/fig_legacy_diagnostics.pdf
figures/fig_qualitative_template.pdf
```

Overleaf Menu:

- Compiler: **pdfLaTeX**
- Main document: **`main.tex`**

Leave `\iclrfinalcopy` commented. Do not add `iclr2027_conference.tex` to the Overleaf project.

GitHub → Overleaf import of the full `hrfont` repo will not compile: the TeX root is `paper/iclr2027_hrfont/`, not the repository root.

## Template audit notes (`71500abd`)

Checked against the official ICLR 2027 zip and [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Passed:

- `\documentclass{article}` + `\usepackage{iclr2027_conference,times}`
- Review line numbers; running head “Under review as a conference paper at ICLR 2027”
- One-paragraph abstract
- Figure captions after figures; table titles before tables; figure widths as fractions of `\linewidth`
- Extra packages (`booktabs`, `microtype`, `xcolor`, …) do not modify the official sty
- No author/institution identifiers in `main.tex`

Not a template failure, but not a finished submission:

- Extra packages and `\small` tables are common; `\resizebox` on Table 2 is the layout item most likely to be challenged
- AI-use statement covers drafting, figure code, and literature checks, and states that AI did not invent measurements. If AI was also used for hypothesis/experiment design or method implementation, align the statement with the [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) before the full-paper deadline

## Changelog

| Date | Change |
|---|---|
| 2026-09-12 | Created tracker from the `71500abd` template audit and the A1–A6 / D1–D8 writing pass. |
