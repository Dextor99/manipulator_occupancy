# 6.5.1 静态障碍补采结果检查

| group | trials | obstacle det | track | risk | minD(m) | pre/post risk max | dominant links | usability |
|---|---:|---:|---:|---:|---:|---:|---|---|
| S1_initial_r01_r03 | 3 | 0.996 | 0.970 | 0.983 | 0.064041 | 0.000 | left_link:229, upperArm_Link:1 | KEEP_AS_TERMINAL_OR_EXCLUDE_FROM_REGION_STATS |
| S1_supplement_r04_r06 | 3 | 0.974 | 0.917 | 0.974 | 0.041872 | 0.000 | upperArm_Link:222 | YES |
| S2_initial_r01_r03 | 3 | 0.991 | 0.965 | 0.991 | 0.041122 | 0.000 | right_link:227 | KEEP_AS_TERMINAL_OR_EXCLUDE_FROM_REGION_STATS |
| S2_supplement_r04_r06 | 3 | 0.974 | 0.948 | 0.974 | 0.04196 | 0.000 | foreArm_Link:154, gripper_base_link:28, shoulder_Link:21, wrist1_Link:20, right_link:2 | YES |
| S3_r01_r03 | 3 | 0.978 | 0.952 | 0.974 | 0.049394 | 0.000 | left_link:220, right_link:5, shoulder_Link:1 | YES |

## 建议采用口径

- S1 肘部/上臂区域：采用 r04-r06，主导最近连杆为 upperArm_Link。
- S2 前臂区域：采用 r04-r06，主导最近连杆为 foreArm_Link，少量 wrist1/gripper/shoulder 由障碍几何和视角引起。
- S3 腕部/末端区域：采用 r01-r03，主导最近连杆为 left_link/right_link。
- S1/S2 的 r01-r03 保留为原始记录，但不建议纳入“区域识别准确性”统计。