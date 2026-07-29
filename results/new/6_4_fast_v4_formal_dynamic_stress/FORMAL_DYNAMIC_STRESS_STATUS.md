# 6.4 Fast v4 D1/D2M Stress Dynamic Status

Run command:

```bash
conda run --no-capture-output -n py310 python -u -m experiments.new.6_4.fast_v4_formal_dynamic_64 \
  --output results/new/6_4_fast_v4_formal_dynamic_stress \
  --scenarios D1 D2M \
  --methods critical_fast_v4 ccro_fast_v4 \
  --risk-band stress \
  --clean
```

Protocol:

- Frozen Fast v4 parameters and thresholds are unchanged.
- Method-independent stress filter: reference dense minimum in `[-0.02, 0.04) m`.
- Stress samples are not used as the main online local-repair success-rate result.

Paper table:

- `results/new/6_4_fast_v4_formal_dynamic_stress/paper/table_6_4_fast_v4_formal_dynamic.md`

## Summary

| scenario | method | n | reference dense min mean / m | reference dense min range / m | candidate dense min mean / m | mean dense gain / m | dense repair | online accept | online P95 / ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | Critical-fast-v4 | 20 | 0.0147 | [-0.0173, 0.0384] | 0.0150 | 0.0003 | 0/20 | 0/20 | 180.7 |
| D1 | CCRO-fast-v4 | 20 | 0.0147 | [-0.0173, 0.0384] | 0.0155 | 0.0008 | 0/20 | 0/20 | 335.0 |
| D2M | Critical-fast-v4 | 20 | 0.0114 | [-0.0166, 0.0397] | 0.0142 | 0.0027 | 0/20 | 0/20 | 182.3 |
| D2M | CCRO-fast-v4 | 20 | 0.0114 | [-0.0166, 0.0397] | 0.0164 | 0.0049 | 0/20 | 0/20 | 335.2 |

## Interpretation

The stress band confirms the operating boundary of single-shot 1 s local repair. These cases require roughly 4--10 cm of additional dense clearance to reach the 0.08 m GT threshold, so they should be discussed as boundary/failure-mode evidence rather than treated as the main 6.4 dynamic validation.
