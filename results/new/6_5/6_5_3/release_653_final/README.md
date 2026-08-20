# 6.5.3 final experimental release

This directory freezes the 6.5.3 evidence package for paper writing. It is a
data-and-documentation release; no robot or camera code is changed here.

## Evidence cases

- `r27`: representative dynamic bypass and terminal recovery; goal reached.
- `r28`: representative rolling/local recovery; goal reached.
- `r29`: stress case, fail-closed hold under a more difficult execution state.
- `r30`: terminal execution stopped by operator-intervention safety handling.

The primary real-robot claims use r27 and r28. r29/r30 are retained as safety
and boundary evidence, not as additional success trials.

## Reproducible artifacts

- Raw trial archives: `../d2_approach_hold_complete_live/r27` through `r30`.
- Final figures and tables: `../paper_figures`.
- Figure generator: `../../../../experiments/new/6_5/6_5_3/plot_653_paper_figures.py`.
- Numeric extraction and interpretation: `paper_numbers.md`.

The experiment implementation is frozen at the Git commit recorded in
`manifest.json`. Subsequent changes should be limited to plotting, data
formatting, and manuscript text.
