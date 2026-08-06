# 6.5.2 Tabletop Planar Motion Revision

## Why The Previous Reference Is Rejected

The historical `config/ccro_stage2.yaml` reference is a generic joint-space
NUBS trajectory:

```text
q_start = [0.0, -0.60, 1.40, 0.0, 1.00, 0.0]
q_goal  = [1.0,  0.10, 0.50, 0.8, 0.30, -0.70]
```

It changes shoulder, elbow and wrist pitch joints substantially.  In the real
tabletop setup this creates a nodding motion toward the desk plane, which is
not an acceptable 6.5.2 hardware trajectory.

Therefore `run_652_static_avoidance.py` now rejects this non-planar reference
by default.  It can only be used for offline debugging with:

```bash
--allow-nonplanar-reference
```

Do not use that flag for real tabletop execution.

## Revised 6.5.2 Motion Principle

The real static-obstacle execution should use a tabletop planar path:

```text
current safe TCP pose
-> horizontal straight reference path at fixed TCP height
-> static obstacle detected before execution
-> horizontal curve/waypoint detour around obstacle
-> low-speed Cartesian execution after validation
```

Required constraints:

| Item | Requirement |
|---|---|
| TCP height | Approximately constant; no downward nodding toward the table. |
| TCP attitude | Approximately constant unless a small wrist change is needed. |
| Joint 2/3/4/5 motion | Small and bounded; no large shoulder/elbow pitch swing. |
| Path type | Straight line for reference; smooth planar curve or via-point path for avoidance. |
| Execution | AUBO supported Cartesian line/waypoint interface, not Python point-by-point joint streaming. |

## Practical Experiment Redesign

For 6.5.2, use the trajectory concept already proven safe in
`robot/safety_guided_motion.py`:

```text
fixed current height and orientation
move in table plane along Y or X-Y
```

Suggested scenes:

| Scene | Reference | Candidate |
|---|---|---|
| R-S1 | Horizontal straight line near a low side obstacle. | Same start/goal with one lateral via point around the obstacle. |
| R-S2 | Horizontal straight line through central high tabletop obstacle region. | Wider or higher-side planar detour while preserving TCP height. |

## How Avoidance Should Be Implemented

For tabletop execution, avoidance should be generated in Cartesian waypoint
space first:

```text
P0 = current TCP pose
P2 = target TCP pose in same horizontal plane
P1 = automatically selected side via point around observed obstacle footprint
```

The candidate path is then a smooth planar curve through `P0-P1-P2`, while the
reference path is the straight segment `P0-P2`.

The current repository can safely plan and visualize this, but a formal real
execution still needs a bounded AUBO Cartesian waypoint executor.  It should
use the robot controller's supported Cartesian motion commands and monitor:

```text
camera timeout
robot state timeout
obstacle movement
tracking error
online observed clearance
manual emergency stop
```

## Immediate Next Step

Before any robot motion:

1. Generate a planar Cartesian reference/candidate preview.
2. Render the URDF/tabletop safety preview.
3. Confirm the path is above the desk and does not nod.
4. Only then implement a guarded low-speed Cartesian waypoint executor.

This replaces the old joint-space `ccro_stage2.yaml` reference for 6.5.2 real
hardware trials.
