# 6.4 Final Paper Package

This package indexes the final 6.4 results after freezing Fast CCRO-NUBS v4.

## Paper Structure

Recommended section structure:

```text
6.4 Dynamic local repair
6.4.1 Setup and admissible-risk protocol
6.4.2 Capability boundary on G1 risk bands
6.4.3 D1/D2 admissible dynamic validation
6.4.4 Stress boundary and limitations
```

Recommended claim boundary:

> Fast CCRO-NUBS targets dynamic obstacle local trajectory repair rather than arbitrary deep collision recovery. Under method-independent admissible-risk sampling, it improves dense whole-body clearance and online acceptance over the Critical-point active-set baseline, especially in D1 whole-body crossing scenarios. Stress-band results show the applicability boundary of single-shot 1 s local repair.

Avoid claiming universal real-time dynamic obstacle avoidance.

## Main Evidence

G1 capability boundary:

- `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_main_performance.md`
- `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_clearance_gain.md`
- `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_runtime_decomposition.md`
- `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_reference_vs_repaired.md`

D1/D2 admissible dynamic validation:

- `results/new/6_4_fast_v4_formal_dynamic_admissible/paper/table_6_4_fast_v4_dynamic_performance.md`
- `results/new/6_4_fast_v4_formal_dynamic_admissible/paper/table_6_4_fast_v4_dynamic_clearance_gain.md`
- `results/new/6_4_fast_v4_formal_dynamic_admissible/paper/table_6_4_fast_v4_dynamic_runtime.md`
- `results/new/6_4_fast_v4_formal_dynamic_admissible/paper/table_6_4_fast_v4_dynamic_representative_cases.md`
- `results/new/6_4_fast_v4_formal_dynamic_admissible/FORMAL_DYNAMIC_ADMISSIBLE_STATUS.md`

Stress and boundary evidence:

- `results/new/6_4_fast_v4_formal_dynamic_stress/paper/table_6_4_fast_v4_dynamic_performance.md`
- `results/new/6_4_fast_v4_formal_dynamic_stress/paper/table_6_4_fast_v4_dynamic_clearance_gain.md`
- `results/new/6_4_fast_v4_formal_dynamic_stress/FORMAL_DYNAMIC_STRESS_STATUS.md`
- `results/new/6_4_fast_v4_formal_dynamic/FORMAL_DYNAMIC_STATUS.md`

## Key Numbers

G1-main:

| Method | Dense repair | Online acceptance | Verified safety | P95 online |
|---|---:|---:|---:|---:|
| Critical-fast-v4 | 8/10 | 3/10 | 3/3 | 123.4 ms |
| CCRO-fast-v4 | 9/10 | 9/10 | 9/9 | 130.3 ms |

D1 admissible:

| Method | Dense repair | Online acceptance | Verified safety | Mean Delta D | P95 online |
|---|---:|---:|---:|---:|---:|
| Critical-fast-v4 | 4/20 | 2/20 | 2/2 | 0.0063 m | 162.7 ms |
| CCRO-fast-v4 | 10/20 | 5/20 | 5/5 | 0.0120 m | 135.3 ms |

D2M admissible:

| Method | Dense repair | Online acceptance | Verified safety | Mean Delta D | P95 online |
|---|---:|---:|---:|---:|---:|
| Critical-fast-v4 | 12/20 | 9/20 | 9/9 | 0.0171 m | 127.7 ms |
| CCRO-fast-v4 | 13/20 | 12/20 | 12/12 | 0.0189 m | 138.0 ms |

Stress band:

- D1 and D2M stress cases produce 0/20 dense repair and 0/20 online acceptance for both methods.
- Use this only as boundary/limitation evidence.
