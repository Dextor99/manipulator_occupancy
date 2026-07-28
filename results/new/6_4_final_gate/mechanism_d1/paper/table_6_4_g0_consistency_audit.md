# 6.4 G0 consistency audit

This audit is a pre-formal implementation check. It must not be used to tune individual failed trials.

## Candidate Funnel

| scenario | method | triggers | finished | within budget | converged | submit continuous | submit online-safe | switch continuous | switch online-safe | beneficial | switched | switched+GT safe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| body_crossing_main | CCRO-NUBS | 12 | 12 | 11 | 7 | 12 | 7 | 1 | 2 | 0 | 0 | 0 |

## G0 Checks

- Distance definition: consistent at module level; candidate-level dense artifacts require replay serialization.
- Time alignment: tau prediction error p95 = 2.099 s, deadline misses = 0.
- Bridge/candidate split: bridge GT mean = 0.149 m, candidate submit mean = 0.157 m, candidate switch mean = 0.206 m.
- Medium/dense trial recheck: medium-minus-dense p95 = 0.001 m; accepted-online/GT-violation trials = 0.

## Decision

- Formal full run: no.
- Reason: Run candidate replay and a simple feasible validation set before any new closed-loop formal test.
