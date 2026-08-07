# r10 Route-Locked Candidate Repair Report

本次修复针对 r10 中“高障碍场景仍选择大幅上方绕行”的问题。r10 原始参考轨迹在真实障碍点云下最小净空约为 `0.0007 m`，明显低于 `0.08 m` 安全阈值，因此该试次本身是有效的静态避障规划样本。

## 修复内容

1. 将多初值生成改为多路径族生成：`free`、`base_side`、`outer_side`、`overpass`。
2. 对 `base_side` 和 `outer_side` 加入路径族保持约束：障碍近区必须位于指定侧向走廊，并且 TCP 高度不能大幅偏离参考轨迹。
3. 对 `overpass` 加入路径族几何一致性验收：只有被 route audit 判定为 `true_overpass` 的候选才能作为 overpass 家族通过。
4. route audit 使用 TCP/tool 核心投影进行路径类别判断，整机 mesh 继续用于 dense 安全验收和诊断。
5. hard feasibility 新增 `route_geometry_ok` 与 `route_family_preserved_if_enabled` 门控。

## r10 修复后结果

选择报告：

`candidate_selection_layered_fixed/candidate_selection_summary.json`

最终状态：

- `status`: `EXECUTABLE_CANDIDATE_SELECTED`
- `selected_execution_candidate`: `base_side`
- `feasible_candidate_count`: `2`
- `near_best_time_candidate_count`: `2`

| Candidate | Hard feasible | Route class | Family ok | Dense min / m | T_req / s | TCP path / m | Joint path / rad | Jerk | Max TCP z dev / m | Reject reasons |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base_side` | true | lateral | true | 0.1098 | 8.0 | 0.8395 | 2.3599 | 0.3694 | 0.0500 | - |
| `seed_free` | true | hybrid_vertical_lateral | true | 0.1139 | 8.0 | 1.0450 | 1.9963 | 0.1653 | 0.2941 | - |
| `overpass` | false | hybrid_vertical_lateral | false | 0.1139 | 8.0 | 1.0450 | 1.9963 | 0.1653 | 0.2941 | route_family_not_preserved |
| `outer_side` | false | lateral | false | 0.0005 | 8.0 | 3.1874 | 5.8615 | 27.8390 | 0.0555 | dense_distance_below_acceptance, route_family_not_preserved |

## 结论

修复后，r10 不再把“由 overpass 初值收敛回混合垂直路径”的候选当作有效 overpass 家族；侧向候选也不再允许在优化中偷偷变成大幅抬升轨迹。由于 `base_side` 与 `seed_free` 执行时间相同，选择器按 TCP 三维路径长度优先，选出 `base_side`，其 TCP 路径长度 `0.8395 m` 明显短于 `seed_free` 的 `1.0450 m`，并且 dense 最小净空 `0.1098 m` 满足安全阈值。

## 图像输出

Top view 已生成：

- `figures/r10_seed_free_top_view.png`
- `figures/r10_base_side_top_view.png`
- `figures/r10_outer_side_top_view.png`
- `figures/r10_overpass_top_view.png`

执行前仍需人工检查 `base_side/figures/ccro_nubs_urdf_pose_sequence.png`、`base_side/figures/joint_trajectory_preview.png` 和现场障碍物是否保持不变。
