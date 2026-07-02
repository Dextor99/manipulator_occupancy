# E1 Experiment Framework

## Reuse Decision

| Part | Existing file/result | Decision |
|---|---|---|
| Object-level spatiotemporal occupancy | `experiments/exp_62_sim_occupancy_risk.py`, `data/results/ch6_2_sim` | Reuse implementation; rerun to 20 trials for formal E1. |
| Whole-body risk distance | `experiments/exp_63_sim_risk_distance.py`, `data/results/ch6_3_sim` | Reuse as internal evidence/ablation; current method set is not the final external APF baseline. |
| Real RGB-D occupancy replay | `data/results/ch6_2_real` | Reuse as real-system support figure/table. |
| Real whole-body risk replay | `data/results/ch6_3_real` | Reuse as real-system support figure/table. |
| Critical-point APF risk | Not implemented yet | Add new script before final E1 main comparison. |

## Formal E1 Outputs

| Output | Source |
|---|---|
| `occupancy_sim/table_6_2_sim.md` | formal R3/R4 object-level occupancy comparison |
| `whole_body_internal/table_6_3_sim.md` | R1/R2/R4 internal whole-body evidence |
| `real_support/` | copied real replay summaries and figures |
| `critical_point_apf/` | pending external-style APF whole-body baseline |

## Commands

Run the formal object-level occupancy comparison:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/exp_62_sim_occupancy_risk.py \
  --scenes static_safe,approach,crossing,leave \
  --trials 20 \
  --plot \
  --output data/results/ch6_e1_e5/E1_occupancy_risk_final/occupancy_sim
```

Run or refresh the whole-body internal evidence:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/exp_63_sim_risk_distance.py \
  --config config/ccro_stage2.yaml \
  --scenes ee_near,body_near,dynamic_future \
  --trials 30 \
  --output data/results/ch6_e1_e5/E1_occupancy_risk_final/whole_body_internal
```

Copy real replay support:

```bash
mkdir -p data/results/ch6_e1_e5/E1_occupancy_risk_final/real_support
cp -r data/results/ch6_2_real data/results/ch6_e1_e5/E1_occupancy_risk_final/real_support/
cp -r data/results/ch6_3_real data/results/ch6_e1_e5/E1_occupancy_risk_final/real_support/
```

## Missing Work

Add `experiments/exp_63_critical_point_apf.py`:

1. Select 1-3 critical points per link from `RobotSurfaceModel`.
2. Compute nearest obstacle distance for those sparse points.
3. Convert distance to APF-style risk or threshold trigger.
4. Compare against `Ours-CCRO` on the same `ee_near`, `body_near`, and `dynamic_future` scenes.
5. Output `detection rate`, `D_min`, `active link`, and `T_risk`.
