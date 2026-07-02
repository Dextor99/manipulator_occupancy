# E1 Status

## Completed

| Block | Output | Status |
|---|---|---|
| Object-level occupancy and STRO warning | `occupancy_sim/` | Completed with 20 trials per scene. |
| Whole-body internal risk evidence | `whole_body_internal/` | Completed with 30 trials per scene. |
| Real RGB-D replay support | `real_support/ch6_2_real`, `real_support/ch6_3_real` | Copied from existing results. |
| Critical-point APF baseline | `critical_point_apf/` | Completed with 30 trials per scene. |
| Final E1 paper tables and figures | `final/` | Completed. |

## Main Results So Far

| Scenario | Useful observation |
|---|---|
| `approach` | Ours-STRO gives longer warning lead time than current-frame and voxel methods, with higher conservative false-trigger time. |
| `crossing` | Ours-STRO gives the strongest dynamic lead time; report the false-trigger trade-off honestly. |
| `body_near` | EEF-only misses middle-link risks; Body-current and Ours-CCRO detect current body risk. |
| `dynamic_future` | Body-current misses future risk; Ours-CCRO detects the future-risk cases. |

## Can Be Used Directly

| Material | Use in thesis |
|---|---|
| `occupancy_sim/table_6_2_sim.md` | E1 main table for current/voxel/OctoMap-like/Ours-STRO. |
| `occupancy_sim/fig_approach.png`, `occupancy_sim/fig_crossing.png` | E1 dynamic warning curves. |
| `whole_body_internal/table_6_3_sim.md` | E1 auxiliary evidence or E5 ablation. |
| `real_support/ch6_2_real/*`, `real_support/ch6_3_real/*` | Real replay support figures/tables. |

## Still Needed

| Missing item | Why it is needed |
|---|---|
| None for the current E1 framework | E1 now has occupancy, sparse critical-point/APF, Ours-CCRO mesh, and real replay support outputs. |

## Exact Commands Already Run

```bash
PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_62_sim_occupancy_risk.py \
  --scenes static_safe,approach,crossing,leave \
  --trials 20 \
  --plot \
  --output data/results/ch6_e1_e5/E1_occupancy_risk_final/occupancy_sim

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_63_sim_risk_distance.py \
  --config config/ccro_stage2.yaml \
  --scenes ee_near,body_near,dynamic_future \
  --trials 30 \
  --output data/results/ch6_e1_e5/E1_occupancy_risk_final/whole_body_internal

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/exp_63_critical_point_apf.py \
  --config config/ccro_stage2.yaml \
  --scenes ee_near,body_near,dynamic_future \
  --trials 30 \
  --plot \
  --output data/results/ch6_e1_e5/E1_occupancy_risk_final/critical_point_apf

PYTHONPATH=. /home/hzy/miniconda3/envs/py310/bin/python experiments/summarize_e1_final.py \
  --root data/results/ch6_e1_e5/E1_occupancy_risk_final
```

## Final Outputs

Use these for the E1 thesis section:

```text
data/results/ch6_e1_e5/E1_occupancy_risk_final/final/table_E1_occupancy_final.md
data/results/ch6_e1_e5/E1_occupancy_risk_final/final/table_E1_whole_body_apf_final.md
data/results/ch6_e1_e5/E1_occupancy_risk_final/final/figures/fig_E1_dynamic_warning.png
data/results/ch6_e1_e5/E1_occupancy_risk_final/final/figures/fig_E1_whole_body_apf.png
```
