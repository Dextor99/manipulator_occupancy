# 6.5.1 静态障碍感知结果可用性检查

| trial | phase | frames | det | track | risk | minD(m) | top links | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| S1_static_elbow_r01 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S1_static_elbow_r01 | obstacle | 77 | 0.987 | 0.961 | 0.974 | 0.07366 | left_link:76 | OK |
| S1_static_elbow_r01 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S1_static_elbow_r02 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S1_static_elbow_r02 | obstacle | 77 | 1.000 | 0.974 | 0.987 | 0.064041 | left_link:77 | OK |
| S1_static_elbow_r02 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S1_static_elbow_r03 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S1_static_elbow_r03 | obstacle | 77 | 1.000 | 0.974 | 0.987 | 0.075021 | left_link:76, upperArm_Link:1 | OK |
| S1_static_elbow_r03 | post_removed | 23 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S2_static_forearm_r01 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S2_static_forearm_r01 | obstacle | 76 | 1.000 | 0.974 | 1.000 | 0.041636 | right_link:76 | OK |
| S2_static_forearm_r01 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S2_static_forearm_r02 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S2_static_forearm_r02 | obstacle | 76 | 0.974 | 0.947 | 0.974 | 0.044575 | right_link:74 | OK |
| S2_static_forearm_r02 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S2_static_forearm_r03 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S2_static_forearm_r03 | obstacle | 77 | 1.000 | 0.974 | 1.000 | 0.041122 | right_link:77 | OK |
| S2_static_forearm_r03 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S3_static_wrist_r01 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S3_static_wrist_r01 | obstacle | 77 | 0.974 | 0.948 | 0.974 | 0.052228 | left_link:71, right_link:4 | OK |
| S3_static_wrist_r01 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S3_static_wrist_r02 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S3_static_wrist_r02 | obstacle | 77 | 0.974 | 0.948 | 0.974 | 0.049394 | left_link:75 | OK |
| S3_static_wrist_r02 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S3_static_wrist_r03 | pre_empty | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |
| S3_static_wrist_r03 | obstacle | 77 | 0.987 | 0.961 | 0.974 | 0.052297 | left_link:74, right_link:1, shoulder_Link:1 | OK |
| S3_static_wrist_r03 | post_removed | 24 | 0.000 | 0.000 | 0.000 | - | - | OK |

## 场景汇总

| scenario | obstacle det mean | track mean | risk mean | min distance | pre risk max | post risk max | dominant links |
|---|---:|---:|---:|---:|---:|---:|---|
| S1 | 0.996 | 0.970 | 0.983 | 0.064041 | 0.000 | 0.000 | left_link:229, upperArm_Link:1 |
| S2 | 0.991 | 0.965 | 0.991 | 0.041122 | 0.000 | 0.000 | right_link:227 |
| S3 | 0.978 | 0.952 | 0.974 | 0.049394 | 0.000 | 0.000 | left_link:220, right_link:5, shoulder_Link:1 |

## 判定

- 可用于证明：真实 RGB-D 下静态障碍可被稳定检测、跟踪，移除后可恢复为空场。
- 需要谨慎：S1/S2/S3 的最近连杆多数为 left_link/right_link，不能直接写成肘部/前臂/腕部分区均已被正确识别。
- 建议：保留这组结果作为“末端附近静态障碍三重复”或“静态障碍检测烟雾/正式初版”；若论文需要三种身体区域证据，需要补采 S1/S2，使障碍物真正靠近 upperArm/foreArm/wrist1。