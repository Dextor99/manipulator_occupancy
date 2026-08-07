# R10 Static Obstacle Precheck

## Status

R10 capture is complete and the perception data are usable, but this trial is not suitable as a formal static-avoidance execution trial.

Reason: the original reference trajectory is already safe under the observed obstacle model.

## Key Metrics

| item | value |
| --- | ---: |
| obstacle points | 2073 |
| obstacle center / m | [0.6998, -0.0051, 0.4129] |
| obstacle bbox min / m | [0.5248, -0.0719, 0.2591] |
| obstacle bbox max / m | [0.7696, 0.1254, 0.5537] |
| table z / m | 0.2778 |
| robust obstacle top p99 / m | 0.5498 |
| reference min obstacle clearance / m | 0.4332 |
| candidate preview min obstacle clearance / m | 0.5181 |
| safety threshold / m | 0.0800 |
| reference risky | false |

## Interpretation

The obstacle was detected stably and the point cloud model is valid. However, the obstacle is far from the planned TCP reference path in the current coordinate frame. The reference trajectory minimum clearance is 0.4332 m, which is far above the 0.08 m safety threshold.

Therefore, R10 does not demonstrate "reference trajectory is unsafe and CCRO-NUBS repairs it". It should not be counted as a formal 6.5.2 static avoidance trial.

## Recommended Action

Keep this result as a no-risk placement check. For the next formal trial, move the obstacle closer to the actual reference path so that:

- the obstacle remains stable and visible to RealSense;
- start and goal configurations remain safe;
- the reference trajectory minimum clearance is below 0.08 m, or at least close to the risk threshold;
- the obstacle does not physically collide with the stationary robot at the start.

Useful existing figures:

- `figures/top_view_tcp_paths.png`
- `figures/clearance_curves.png`
- `figures/obstacle_model_pointcloud.png`
- `figures/reference_pose_sequence.png`
- `figures/candidate_pose_sequence.png`
