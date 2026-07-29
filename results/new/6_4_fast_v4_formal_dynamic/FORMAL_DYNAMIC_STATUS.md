# 6.4 Fast v4 Formal Compact Dynamic Run Status

Run command:

```bash
conda run --no-capture-output -n py310 python -u -m experiments.new.6_4.fast_v4_formal_dynamic_64 \
  --output results/new/6_4_fast_v4_formal_dynamic \
  --scenarios D1 D2M \
  --methods critical_fast_v4 ccro_fast_v4 \
  --clean
```

Paper table:

- `results/new/6_4_fast_v4_formal_dynamic/paper/table_6_4_fast_v4_formal_dynamic.md`

## Result

The compact D1/D2M run completed successfully, but it should not be used as the main paper success-rate result.

The generated unfiltered compact dynamic cases are deep-risk stress cases:

| scenario | method | n | reference dense min mean / m | reference dense min range / m | candidate dense min mean / m | dense repair | online accept |
|---|---|---:|---:|---:|---:|---:|---:|
| D1 | Critical-fast-v4 | 20 | -0.0766 | [-0.1037, -0.0190] | -0.0743 | 0/20 | 0/20 |
| D1 | CCRO-fast-v4 | 20 | -0.0766 | [-0.1037, -0.0190] | -0.0745 | 0/20 | 0/20 |
| D2M | Critical-fast-v4 | 20 | -0.0592 | [-0.1078, -0.0128] | -0.0566 | 0/20 | 0/20 |
| D2M | CCRO-fast-v4 | 20 | -0.0592 | [-0.1078, -0.0128] | -0.0513 | 0/20 | 0/20 |

All dense and online failures are distance-gate failures. Acceleration gates pass, so the failure mode is geometric infeasibility under deep initial penetration rather than dynamic-limit violation.

## Interpretation

This run is useful as a stress/boundary audit: a single-shot 1 s local QP repair is not expected to recover 6--10 cm penetrations to the dense GT threshold of 0.08 m. The result does not contradict the frozen G1-band capability curve, where CCRO-fast-v4 repairs shallow-risk cases within the realtime budget.

For the paper, keep the G1-band study as the main publishable evidence and cite this compact D1/D2M run only as a deep-risk boundary result unless a method-independent admissible-risk D1/D2M generator is added.
