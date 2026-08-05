# 6.5.1 静态障碍结果检查

| group | trials | obstacle det | track | risk | minD(m) | pre/post risk max | dominant links | usability |
|---|---:|---:|---:|---:|---:|---:|---|---|
| S1_elbow_upperArm_r04_r06 | 3 | 0.974 | 0.917 | 0.974 | 0.041872 | 0.000 | upperArm_Link:222 | YES |
| S2_forearm_r04_r06 | 3 | 0.974 | 0.948 | 0.974 | 0.04196 | 0.000 | foreArm_Link:154, gripper_base_link:28, shoulder_Link:21, wrist1_Link:20, right_link:2 | YES |
| S3_wrist_end_r01_r03 | 3 | 0.978 | 0.952 | 0.974 | 0.049394 | 0.000 | left_link:220, right_link:5, shoulder_Link:1 | YES |

## 说明

- S1/S2 早期靠近末端的 r01-r03 已从正式 `trials/` 目录删除。
- 当前静态正式目录只保留 S1 r04-r06、S2 r04-r06 和 S3 r01-r03。