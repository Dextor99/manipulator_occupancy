# 6.5.3 Dynamic Obstacle Fast CCRO-NUBS Real-System Trial

This folder contains the first staged implementation for Section 6.5.3.

The experiment reuses the safe 6.5.2 tabletop reference:

- home joints: `[0, 0, 90, 0, 90, 0] deg`
- TCP X: home TCP + `0.10 m`
- TCP Y: `+0.40 m -> -0.40 m`
- TCP Z / orientation: fixed from the home TCP pose

## Modes

`run_653_dynamic_repair_trial.py` supports:

- `shadow`: RealSense + AUBO feedback only, no robot command.
- `moving-shadow-stop`: command the low-speed reference line, trigger Fast
  CCRO-NUBS, then stop; no online switch yet.
- `live-execute`: intentionally blocked in this first implementation.

Use `moving-shadow-stop` as the required pilot before enabling true online
trajectory switching.

## 1. Optional Reference Recording

Dry run:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/prepare_653_reference.py \
  --output results/new/6_5/6_5_3/reference_xp10_line
```

Real low-speed reference recording:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/prepare_653_reference.py \
  --execute \
  --operator-phrase CCRO_653_REFERENCE_APPROVED \
  --reference-shape line \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.025 \
  --line-acc-m-s2 0.06 \
  --record-duration-s 40 \
  --output results/new/6_5/6_5_3/reference_xp10_line
```

## 2. D1/D2 Shadow Trial

This mode does not move the robot:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 \
  --repeat 1 \
  --mode shadow \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

For D2:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D2 \
  --repeat 1 \
  --mode shadow \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

## 3. Moving Shadow With Stop

This commands the low-speed reference line and stops after risk trigger. It
generates and validates a Fast CCRO-NUBS candidate, but does not switch to it.

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 \
  --repeat 1 \
  --mode moving-shadow-stop \
  --operator-phrase CCRO_653_DYNAMIC_SHADOW_APPROVED \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.020 \
  --line-acc-m-s2 0.05 \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

## Outputs

Each trial writes:

- `frames.csv`
- `summary.json`
- `candidate/candidate_summary.json` if triggered
- `candidate/fast_ccro_nubs_candidate.csv` if triggered

Plot one trial:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/plot_653_dynamic_trial.py \
  --trial-dir results/new/6_5/6_5_3/dynamic_repair_pilot/trials/D1_crossing_body_r01
```

Summarize a result folder:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/summarize_653_dynamic_results.py \
  --input results/new/6_5/6_5_3/dynamic_repair_pilot
```

Official 6.5.3 live switching should only be enabled after D1/D2
`moving-shadow-stop` logs show stable trigger timing and accepted candidates.

## Stop Interface Validation

After any contact in moving dynamic pilots, do not continue obstacle trials.
First validate the controller stop API in an empty workspace:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/validate_653_stop_interface.py \
  --execute \
  --operator-phrase CCRO_653_STOP_VALIDATION_APPROVED \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.010 \
  --line-acc-m-s2 0.025 \
  --stop-after-s 3.0 \
  --record-after-stop-s 4.0 \
  --output results/new/6_5/6_5_3/stop_interface_validation/r01
```
