dry_run_accepted: True
unsafe_switch_blocked: True
real_robot_complete: False
claim_boundary: Use as real sensing/replay + software readiness evidence. Do not claim completed real robot online trajectory switching until P7 pending checks pass.

Quality checks:
- PASS: 时空预测消融 - Full T_lead=5.1286, w/o Temporal=4.7356
- PASS: 排斥向量消融 - Full D_min_ref=0.1968, w/o Rep D_min_ref=0.0561, w/o Rep T_viol=0.4317
- PASS: 安全滤波消融 - Full D_min_ref=0.1968, w/o Filter D_min_ref=0.1041
- PASS: 试次数量 - 4.5 闭环消融统计 n_trials=10
- NOTE: 实时性日志完整性 - 当前 4.5 日志只对控制器/排斥计算段计时；感知预处理、解耦、聚类、跟踪、预测列为 0。
- NOTE: 控制段实时性目标 - T_ctrl^95=199.8850 ms，目标 < 20 ms。
- NOTE: 端到端实时性目标 - T_frame^95=199.8850 ms；但当前缺少完整感知模块计时。
