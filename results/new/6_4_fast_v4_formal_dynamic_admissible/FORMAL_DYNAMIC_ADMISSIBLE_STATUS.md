# 6.4 Fast v4 D1/D2M Admissible Dynamic Status

Run command:

```bash
conda run --no-capture-output -n py310 python -u -m experiments.new.6_4.fast_v4_formal_dynamic_64 \
  --output results/new/6_4_fast_v4_formal_dynamic_admissible \
  --scenarios D1 D2M \
  --methods critical_fast_v4 ccro_fast_v4 \
  --risk-band admissible \
  --clean
```

Protocol:

- Frozen Fast v4 parameters: `target_clearance=0.095`, one QP iteration, `clearance_reward=0.0`.
- Safety thresholds unchanged: dense GT `D_STOP=0.08 m`, online acceptance `D_ONLINE_ACCEPT=0.09 m`.
- Method-independent sampling filter:
  - reference dense minimum in `[0.04, 0.08) m`;
  - dense active risk must be controllable by the 1 s local NUBS repair variables.

Paper table:

- `results/new/6_4_fast_v4_formal_dynamic_admissible/paper/table_6_4_fast_v4_formal_dynamic.md`

## Summary

| scenario | method | n | reference dense min mean / m | reference dense min range / m | candidate dense min mean / m | mean dense gain / m | dense repair | online accept | online P95 / ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | Critical-fast-v4 | 20 | 0.0636 | [0.0410, 0.0786] | 0.0699 | 0.0063 | 4/20 | 2/20 | 162.7 |
| D1 | CCRO-fast-v4 | 20 | 0.0636 | [0.0410, 0.0786] | 0.0755 | 0.0120 | 10/20 | 5/20 | 135.3 |
| D2M | Critical-fast-v4 | 20 | 0.0618 | [0.0401, 0.0796] | 0.0789 | 0.0171 | 12/20 | 9/20 | 127.7 |
| D2M | CCRO-fast-v4 | 20 | 0.0618 | [0.0401, 0.0796] | 0.0807 | 0.0189 | 13/20 | 12/20 | 138.0 |

## Interpretation

This is the current formal D1/D2M dynamic extension result. It supports a bounded claim: Fast CCRO-NUBS v4 improves dense whole-body clearance and outperforms the Critical-point active-set baseline under the same admissible dynamic protocol, especially in D1 crossing and D2M online acceptance.

The result should be reported alongside the G1-band capability curve, not as a universal dynamic obstacle avoidance success rate. D1 remains the harder crossing case: CCRO repairs 10/20 dense cases and online-accepts 5/20, while D2M reaches 13/20 dense repair and 12/20 online acceptance.
