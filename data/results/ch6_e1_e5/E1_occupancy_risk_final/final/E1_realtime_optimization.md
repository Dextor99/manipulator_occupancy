# E1 Whole-Body Risk Runtime Optimization

## Why the Original Mesh Risk Was Slow

The first Critical-point APF comparison reported about `158 ms` for Ours-CCRO Mesh and about `4.2 ms` for Critical-point APF.  The gap came from both algorithmic scope and implementation details.

| Cause | Explanation |
|---|---|
| Representation size | Critical-point APF checks only about 30 sparse points; mesh risk checks thousands of robot surface points. |
| Future obstacle size | Ours-CCRO evaluates current and predicted future obstacle spheres, so the obstacle cloud is much larger than a current-frame check. |
| Old query direction | The previous helper rebuilt a robot KDTree for every link and repeatedly queried all future obstacle points. |
| Dense validation mode | The original table used dense mesh samples, suitable for offline validation but expensive for a real-time loop. |

## Implemented Optimization

`experiments/exp_63_critical_point_apf.py` now uses `evaluate_mesh_distance_fast()`:

```text
old: for each link, build robot KDTree, query all obstacle points
new: build obstacle KDTree once, query each link surface once
```

This keeps the dense mesh semantics but avoids repeated KDTree construction and repeated full obstacle-cloud scans.

## Runtime After Optimization

| mesh mode | ee_near T_ms | body_near T_ms | dynamic_future T_ms | dynamic_future R_future |
|---|---:|---:|---:|---:|
| dense | 62.9940 | 60.1595 | 26.0806 | 1.0000 |
| medium | 15.8312 | 15.3876 | 8.2584 | 1.0000 |
| coarse | 7.7683 | 7.6716 | 5.1015 | 1.0000 |
| Critical-point APF | 4.2511 | 4.2198 | 4.0831 | 0.7647 |

## Recommended Real-Time Strategy

| Layer | Recommended setting | Role |
|---|---|---|
| High-frequency monitor | coarse mesh risk | Run at control/safety-filter frequency. |
| Normal online risk update | medium mesh risk | Balance coverage and runtime for rolling risk evaluation. |
| Offline validation / final acceptance | dense mesh risk | Use for final dense safety check and thesis reporting. |
| Candidate trajectory optimization | low-frequency trigger | Do not run full CCRO-NUBS optimization every perception frame. |

## Thesis Interpretation

The fair conclusion is not that mesh risk is intrinsically `158 ms`.  The optimized implementation reduces dense runtime substantially, and coarse/medium mesh modes provide near real-time performance while preserving the tested detection rate in E1.  Critical-point APF remains faster because it checks far fewer points, but it misses future-only risks in the dynamic scene.
