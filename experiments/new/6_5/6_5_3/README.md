# 6.5.3 Dynamic Obstacle Fast CCRO-NUBS Real-System Trial

This folder contains the staged implementation for Section 6.5.3. D1 r02-r15
are development logs and are not formal paper evidence.

The experiment reuses the safe 6.5.2 tabletop reference:

- home joints: `[0, 0, 90, 0, 90, 0] deg`
- TCP X: home TCP + `0.10 m`
- TCP Y: `+0.40 m -> -0.40 m`
- TCP Z / orientation: fixed from the home TCP pose

## Modes

`run_653_dynamic_repair_trial.py` supports:

- `shadow`: RealSense + AUBO feedback only, no robot command.
- `moving-shadow-stop`: command the recorded one-way low-speed reference line, trigger Fast
  CCRO-NUBS, then stop; no online switch yet.
- `live-stop-replan-execute`: currently stops and plans but is fail-closed before
  execution until a fresh post-planning RGB-D recheck is implemented.

Use `moving-shadow-stop` as the required pilot before enabling true online
trajectory switching.

The moving modes fail closed unless `--reference-feedback-csv` is supplied.
The same recorded reference drives future STRO evaluation and supplies the
candidate endpoint at `t + H_local`; `q + qd * lookahead` is no longer used.

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
  --reference-feedback-csv results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv \
  --operator-phrase CCRO_653_DYNAMIC_SHADOW_APPROVED \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.020 \
  --line-acc-m-s2 0.05 \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

Before any live candidate execution, obtain three consecutive D1
`moving-shadow-stop` pilots with a stable track ID, plausible tracked radius,
nonzero obstacle speed, `accepted_steps > 0`, at least 3 mm clearance gain,
candidate clearance at least 0.09 m, and no safety-gate failure. D2 is optional
until five formal D1 trials have completed.

The stop interface is frozen based on `stop_interface_validation/r03`; do not
repeat that validation merely to tune Fast-repair thresholds. Formal settings
remain `replan=0.14 m`, `H_local=1.0 s`, `online_accept=0.09 m`, and
`fast_budget=150 ms`.

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
