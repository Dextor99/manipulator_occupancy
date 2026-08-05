# D2 动态腕部/末端斜向接近试次质量检查

| trial | dynamic det | track | risk | minD | predMinD | max speed | first pred frame | first current frame | lead frames | post risk | dominant link | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| D2_dynamic_oblique_wrist_approach_r02 | 0.664 | 0.646 | 0.451 | 0.047357 | -0.090713 | 0.097 | 35 | 37 | 2 | 0.000 | wrist1_Link:38, wrist2_Link:21, gripper_base_link:16 | OK |
| D2_dynamic_oblique_wrist_approach_r03 | 0.637 | 0.611 | 0.416 | 0.039915 | -0.018762 | 0.097 | 55 | 58 | 3 | 0.000 | wrist2_Link:28, wrist1_Link:27, gripper_base_link:17 | OK |
| D2_dynamic_oblique_wrist_approach_r04 | 0.637 | 0.619 | 0.451 | 0.044445 | -0.029736 | 0.074 | 58 | 62 | 4 | 0.000 | wrist2_Link:42, wrist1_Link:19, gripper_base_link:11 | OK |
| D2_dynamic_oblique_wrist_approach_r05 | 0.614 | 0.596 | 0.377 | 0.04656 | -0.013169 | 0.089 | 54 | 57 | 3 | 0.000 | wrist2_Link:41, wrist1_Link:17, gripper_base_link:12 | OK |
| D2_dynamic_oblique_wrist_approach_r06 | 0.615 | 0.596 | 0.477 | 0.055039 | -0.015781 | 0.079 | 39 | 44 | 5 | 0.000 | wrist2_Link:42, gripper_base_link:20, wrist1_Link:5 | OK |

## 最终采用口径

- D2 正式统计采用当前目录中的 r02-r06，共 5 组。
- 原 D2 r01 因主导最近连杆为 upperArm_Link、实际更符合 D1，已从 `trials/` 中删除，避免后续误用。