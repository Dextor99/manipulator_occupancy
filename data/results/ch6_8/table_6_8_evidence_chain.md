| section | claim | evidence | result table | paper-ready | boundary |
| --- | --- | --- | --- | --- | --- |
| 6.2 | 对象级时空占据可支持动态障碍预警 | sim/real occupancy and warning tables | data/results/ch6_2_sim/table_6_2_sim.md; data/results/ch6_2_real/table_6_2_real_warning.md | yes | 真实结果为回放/统计预警，不是实机轨迹切换。 |
| 6.3 | CCRO-NUBS 可在静态近场场景中生成通过 dense 复核的连续轨迹 | static near-field benchmark with frozen instances and independent dense verification | data/results/6_3/paper/table_6_3_static_benchmark.md; data/results/6_3/paper/figure_5_static_trajectory_and_jerk.png | yes | dense feasible rate and 10s budgeted accepted rate are reported separately. |
| 6.3 auxiliary | 全身风险距离能发现末端方法遗漏的中间连杆风险 | body-link sim and real replay counterexamples | data/results/ch6_3_sim/table_6_3_sim.md; data/results/ch6_3_real/table_6_3_real_methods.md | yes | 真实部分为采集序列回放，不发送新控制命令。 |
| 6.4 | CCRO-NUBS 可在静态风险场景中生成通过 dense 复核的连续轨迹 | NUBS internal ablation + MINCO/RRT external baselines | data/results/ch6_4/table_6_4_static_risk.md; data/results/ch6_4/table_6_4_external_baselines.md | yes | MINCO/RRT 为关节空间复现基线，不宣称官方工程直接部署。 |
| 6.5 | 动态障碍下风险触发重规划可提升最小距离并支持安全接管 | virtual closed loop and rolling replanning | data/results/ch6_5/table_6_5_virtual_loop.md; data/results/ch6_5/table_6_5_rolling_replan.md | yes | 主要为虚拟闭环和软件闭环。 |
| 6.6 | 真实系统具备感知回放安全响应和 fail-closed 软件门控 | real replay + P7 dry-run | data/results/ch6_6/table_6_6_real_dynamic_response.md; data/results/ch6_6/table_6_6_readiness_gate.md | yes | 真实在线 NUBS 轨迹切换仍 pending。 |
| 6.7 | 时序预测、全身风险、排斥向量和安全滤波均有独立贡献 | risk and control ablations | data/results/ch6_7/table_6_7_risk_ablation.md; data/results/ch6_7/table_6_7_control_ablation.md | yes | 真实端到端完整分模块计时仍需补非零感知日志。 |
