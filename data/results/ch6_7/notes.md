Consistency notes:
- Simulation and real replay agree that temporal full-body risk is necessary for body-link obstacles.
- Current real timing logs contain controller-side timing but not complete RGB-D preprocessing timing; use 6.6/6.7 timing claims conservatively.
- P5 optimized planner timing is the best current source for online replanning budget.

Quality checks:
- PASS: 时空预测消融 - Full T_lead=5.1286, w/o Temporal=4.7356
- PASS: 排斥向量消融 - Full D_min_ref=0.1968, w/o Rep D_min_ref=0.0561, w/o Rep T_viol=0.4317
- PASS: 安全滤波消融 - Full D_min_ref=0.1968, w/o Filter D_min_ref=0.1041
- PASS: 试次数量 - 4.5 闭环消融统计 n_trials=10
- NOTE: 实时性日志完整性 - 当前 4.5 日志只对控制器/排斥计算段计时；感知预处理、解耦、聚类、跟踪、预测列为 0。
- NOTE: 控制段实时性目标 - T_ctrl^95=199.8850 ms，目标 < 20 ms。
- NOTE: 端到端实时性目标 - T_frame^95=199.8850 ms；但当前缺少完整感知模块计时。
