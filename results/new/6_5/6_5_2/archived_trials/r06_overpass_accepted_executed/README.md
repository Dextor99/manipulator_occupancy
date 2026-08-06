# 6.5.2 R-S1 r06 归档说明

本目录冻结 r06 实验数据，供后续论文制图、结果复核和与新障碍布局实验对比使用。原始路径仍保留，归档采用复制方式，不影响后续脚本读取。

## 实验定位

- 场景：R-S1 侧向桌面障碍，轨迹整体相对原基准向 X 正方向平移 0.10 m。
- 规划方法：joint-space CCRO-NUBS，不是 Cartesian Bezier/B-spline。
- r06 原始候选：通过 dense verifier，并完成实机避障运动观察；轨迹形态偏向上方绕行。
- flatZ 候选：验证了近似平面绕行具有几何可行性，但优化未收敛，不能作为正式执行候选。

## 核心指标

- 原参考 dense 最小距离：0.000088 m，未通过。
- 原始 CCRO-NUBS 候选状态：PLAN_ACCEPTED，accepted_for_real_execution=True。
- 原始候选 dense 最小距离：0.116845 m。
- 原始候选最近连杆：left_link。
- 原始候选优化时间：10112.26 ms，迭代 47 次。
- flatZ 候选状态：PLAN_REJECTED，拒绝原因：['solver_ok']。
- flatZ dense 最小距离：0.084539 m。
- flatZ TCP Z 范围：0.015406 m。

## 轨迹选择说明

r06 原始 CCRO-NUBS 候选虽然安全并可执行，但因目标函数未惩罚 TCP 高度变化，最终选择了明显的上方绕行。该结果可作为“上绕候选成功执行”的代表，不建议用它单独说明系统总是选择最佳桌面平面绕行。

后续多次实验建议新增候选族选择逻辑：planar-left、planar-right、over、hybrid 先分别生成，再由 dense verifier 过滤，最后按安全裕度、TCP 高度变化、关节变形和轨迹长度排序。

## 归档内容

- `trial/`：r06 感知、点云、规划结果、top view、distance curve、URDF pose sequence。
- `execution_logs/`：r06 各次执行/调速/Offline Track 日志。
- `positioning_logs/`：X+0.10 m 目标捕获与起点复位日志。
- `file_manifest.txt`：归档文件清单。

## 后续注意

- 新实验请使用新的 repeat 编号，例如 r07、r08、r09，避免覆盖 r06。
- 若重新摆放障碍物，应重新采集障碍点云并重新规划，不要复用 r06 的 `obstacle_points.npz`。
- 若使用 Offline Track，速度主要由 `--playback-duration-s` 控制，而不是单纯由 `--joint-velc` 控制。
