# 6.5.3 paper numbers

These values are extracted from the archived JSON/CSV files and are intended
for manuscript tables and captions. Distances are in metres, times in ms.

| Case | Representative behavior | STRO predicted distance | Terminal minimum clearance | Terminal planning time | Outcome |
|---|---|---:|---:|---:|---|
| r27 | Dynamic bypass and recovery | 0.1063 | 0.1073 | 6735.8 | Goal reached |
| r28 | Rolling/local recovery | 0.1287 | 0.1114 | 5566.4 | Goal reached |
| r29 | Stress / fail-closed hold | — | — | — | Safe hold |
| r30 | Terminal execution intervention | — | 0.1055 (authorized plan) | 6778.7 | Safe stop |

The successful local-execution tracking RMSE values are approximately
`0.00128 rad` (r27) and `0.00143 rad` (r28). The final terminal verifier keeps
the production authorization threshold at `0.09 m`; the raw hard guard remains
`0.10 m`, and the STRO trigger distance remains `0.14 m`.

## Recommended paper presentation

Use r27 as the main successful sequence and r28 as the complementary recovery
sequence. Present r29/r30 only as fail-closed boundary/safety evidence. The
simulation study in Section 6.4 supplies statistical comparisons; these real
robot cases establish physical closed-loop behavior rather than a claim of
universal success over all obstacle geometries.
