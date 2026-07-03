# E2 Status

## Current Reusable Results

| Block | Source | Current status | Thesis use |
|---|---|---|---|
| CCRO-NUBS static full-body planning | `reuse/ccro_stage2` | Accepted; scenarios A/B/C all full-body pass dense verification. | Use as our method result. |
| NUBS-base / EEF-risk variants | `reuse/ccro_stage2` | Available. | Use as internal ablation, not main external benchmark. |
| MINCO-base / MINCO-risk | `reuse/ch6_4_external` | Available. | Use MINCO-risk as continuous-trajectory external baseline; MINCO-base as auxiliary. |
| RRT-Connect + smoothing | `reuse/ch6_4_external` | Available with 30 seeds per scenario. | Use as sampling external baseline. |
| Old summary tables | `reuse/ch6_4` | Available. | Use only after wording is corrected for missing classical optimizers. |
| Official TrajOpt/Tesseract baseline | `official_tesseract_trajopt` | Completed through `tesseract-robotics==0.4.0`; solver succeeds on A/B/C. | Use as official package baseline; dense verifier marks it unsafe because point-cloud risk is not injected into Tesseract. |
| CHOMP / TrajOpt / GPMP2 style baselines | `classical_optimizers` | Completed. | Use as lightweight reproduction baselines, not official package bindings. |
| P1/P2/P3 perturbation batch | `perturbation_batch` | Completed with 10 perturbations per scenario under supplemental coarse verification. | Use as robustness statistics; final acceptance table remains dense. |
| Ours fast candidate mode | `ours_fast_mode` | Completed with coarse optimizer mesh and dense validation; P1/P2/P3 all accepted. | Use as runtime-improvement evidence, not as a replacement for the full high-fidelity main result. |
| Weakness/runtime audit | `analysis` | Completed. | Use to discuss disadvantages honestly and frame real-time feasibility by layers. |
| P2 D_min(t) curve | `final/figures/fig_E2_P2_Dmin_curve.png` | Completed. | Use as representative time-series figure for the middle-link near-obstacle case. |
| P2 joint kinematics curve | `final/figures/fig_E2_P2_joint_kinematics.png` | Completed. | Use to support trajectory smoothness with velocity and acceleration norms. |
| P2 J_smooth bar chart | `final/figures/fig_E2_P2_Jsmooth_bar.png` | Completed. | Use to quantify smoothness; unsafe official baseline is hatched and marked. |
| Final E2 table and figure | `final` | Completed. | Use for E2 thesis section; keep official TrajOpt and `*-style` baselines clearly separated. |

## What Can Be Used Directly

| Material | Why usable |
|---|---|
| `reuse/ccro_stage2/table_stage2.md` | Uses the project dense verifier and shows full-body CCRO-NUBS passes A/B/C. |
| `reuse/ch6_4_external/table_6_4_external_baselines.md` | Contains MINCO-risk and RRT-Connect + smoothing under the same verifier. |
| `reuse/ch6_4/table_6_4_rrt_multiseed.md` | RRT already has 30 seeds per scenario, enough for the current reduced benchmark. |
| `official_tesseract_trajopt/table_E2_official_tesseract_trajopt.md` | Contains the official Tesseract/TrajOpt solver result evaluated by the common dense verifier. |
| `classical_optimizers/table_E2_classical_optimizers.md` | Contains CHOMP-style, TrajOpt-style, and GPMP2-style baselines under the same verifier. |
| `perturbation_batch/table_E2_perturbation_batch.md` | Contains 10 perturbed obstacle verifications per scenario for trajectory robustness statistics. |
| `ours_fast_mode/table_stage2.md` | Contains the fast-mode CCRO-NUBS result with dense validation kept unchanged. |
| `analysis/table_E2_ours_runtime_modes.md` | Shows full-vs-fast runtime, D_min, smoothness, and speedup. |
| `analysis/CH6_method_weakness_runtime_review.md` | Summarizes disadvantages and recommended runtime framing. |
| `final/table_E2_static_planning_final.md` | Merges RRT, Official TrajOpt/Tesseract, CHOMP-style, TrajOpt-style, GPMP2-style, MINCO-risk, and Ours CCRO-NUBS. |

## What Should Not Be Used As Main External Comparison

| Material | Reason |
|---|---|
| `NUBS-base` | Internal variant without obstacle risk. |
| `NUBS-EEF-risk` | Internal ablation; useful for showing body-risk necessity. |
| `MINCO-base` | No risk term; useful as auxiliary continuous-trajectory lower baseline. |
| `data/results/ch6_4/missing_external_baselines.md` | Its statement that all external baselines are implemented is inconsistent with the new E2 plan. |

## Remaining Optional Work

| Item | Priority | Supplement plan |
|---|---|---|
| Official CHOMP / GPMP2 package bindings | Optional | Official TrajOpt/Tesseract is complete. Add official MoveIt CHOMP or GPMP2 only if stronger external-package fidelity is required. |
| Full re-optimization on every perturbation | Low | Current perturbation batch re-verifies fixed planned trajectories. Full re-optimization can be added only if a larger statistical benchmark is required. |
| Analytic/contact-gradient implementation | Future engineering | Current full-body optimizer uses finite-difference risk gradients; analytic gradients or active-surface culling would be the next real-time optimization step. |

## Recommended Next Operations

1. Review `final/table_E2_static_planning_final.md` and `final/figures/fig_E2_Dmin_methods.png`.
2. Use `perturbation_batch/table_E2_perturbation_batch.md` as supplemental robustness evidence rather than replacing the dense final table.
3. Use `analysis/table_E2_ours_runtime_modes.md` to show that `Ours-fast` reduces planning time while preserving dense acceptance.
4. Use `final/figures/fig_E2_P2_Dmin_curve.png` for safety-distance evolution, and use `fig_E2_P2_joint_kinematics.png` plus `fig_E2_P2_Jsmooth_bar.png` for motion smoothness.
5. In the thesis wording, call `Official TrajOpt/Tesseract` an official-package baseline, and keep `CHOMP-style`, `TrajOpt-style`, and `GPMP2-style` labels for lightweight reproductions.

## Existing Commands To Reproduce Current Reusable Results

Official TrajOpt/Tesseract requires the official Python wheel and the current wheel needs NumPy 1.x ABI:

```bash
/home/hzy/miniconda3/envs/py310/bin/python -m pip install 'numpy<2' tesseract-robotics
```

```bash
PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_ccro_stage2.py \
  --config config/ccro_stage2.yaml \
  --output data/results/ch6_e1_e5/E2_static_planning_benchmark/ccro_nubs

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_64_external_baselines.py \
  --config config/ccro_stage2.yaml \
  --output data/results/ch6_e1_e5/E2_static_planning_benchmark/external_baselines \
  --rrt-seeds 30

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_64_classical_optimizers.py \
  --config config/ccro_stage2.yaml \
  --output data/results/ch6_e1_e5/E2_static_planning_benchmark/classical_optimizers \
  --max-iterations 16

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_65_official_tesseract_trajopt.py \
  --config config/ccro_stage2.yaml \
  --output data/results/ch6_e1_e5/E2_static_planning_benchmark/official_tesseract_trajopt

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/summarize_e2_robustness.py \
  --root data/results/ch6_e1_e5/E2_static_planning_benchmark \
  --config config/ccro_stage2.yaml \
  --trials 10 \
  --verify-density coarse \
  --verify-time-step 0.1

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_ccro_stage2.py \
  --config config/ccro_stage2_fast.yaml \
  --output data/results/ch6_e1_e5/E2_static_planning_benchmark/ours_fast_mode

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/summarize_ch6_runtime_review.py

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/summarize_e2_final.py \
  --root data/results/ch6_e1_e5/E2_static_planning_benchmark
```

## Final Outputs

```text
data/results/ch6_e1_e5/E2_static_planning_benchmark/final/table_E2_static_planning_final.md
data/results/ch6_e1_e5/E2_static_planning_benchmark/perturbation_batch/table_E2_perturbation_batch.md
data/results/ch6_e1_e5/E2_static_planning_benchmark/analysis/table_E2_ours_runtime_modes.md
data/results/ch6_e1_e5/E2_static_planning_benchmark/analysis/CH6_method_weakness_runtime_review.md
data/results/ch6_e1_e5/E2_static_planning_benchmark/final/E2_final_summary.md
data/results/ch6_e1_e5/E2_static_planning_benchmark/final/figures/fig_E2_Dmin_methods.png
data/results/ch6_e1_e5/E2_static_planning_benchmark/final/figures/fig_E2_P2_Dmin_curve.png
data/results/ch6_e1_e5/E2_static_planning_benchmark/final/figures/fig_E2_P2_joint_kinematics.png
data/results/ch6_e1_e5/E2_static_planning_benchmark/final/figures/fig_E2_P2_Jsmooth_bar.png
```
