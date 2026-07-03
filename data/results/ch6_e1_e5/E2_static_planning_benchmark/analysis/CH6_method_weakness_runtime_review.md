# Chapter 6 Weakness and Runtime Review

## Main Weaknesses To State Honestly

- E1: Ours-STRO has longer warning lead time, but also higher conservative false-trigger time.
  In approach, Ours false-time=0.4571, current-frame=0.0000.
  In crossing, Ours false-time=0.8677, current-frame=0.0036.
- E1: Dense Ours-CCRO mesh risk is slower than Critical-point APF; the advantage is future/full-body detection, not raw per-frame speed.
- E2: Full CCRO-NUBS is not always the highest-clearance or fastest planner.
- E2: The correct advantage is the combination of dense safety acceptance, very low smoothness cost, full-body risk coupling, and stable perturbation results.

## Runtime Framing

- High-frequency layer: occupancy/STRO and coarse or medium mesh risk monitoring.
- Low-frequency layer: CCRO-NUBS candidate generation when risk is triggered.
- Acceptance layer: dense verifier for final safety gate. This can be slower because it is not the continuous control-rate loop.

## Practical Improvement Already Verified

- `config/ccro_stage2_fast.yaml` uses coarse optimizer mesh, fewer risk samples, fewer dynamic samples, and a looser convergence tolerance.
- It keeps dense validation unchanged and passes P1/P2/P3.
- It should be presented as `Ours-fast` or `fast candidate generation`, not as a replacement for the full high-fidelity result.

## Thesis Wording

Use this wording idea: the full method is a conservative high-fidelity planner, while the fast mode is the online candidate generator; both are filtered by the same dense safety gate.
