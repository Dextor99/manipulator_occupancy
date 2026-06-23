# CCRO-NUBS P5：实时性优化与 30 次统计实施计划

## 1. 目标

在不降低 dense 最终复核标准的前提下，将阶段 4 重规划从约 `1.46 s` 降到：

```text
首要目标 planner p95 < 1000 ms
增强目标 planner p95 < 500 ms
control p95 < 20 ms
```

## 2. 本轮优化

1. 优化内部 collision Mesh coarse 点数从 500 降到 300；
2. 每段风险采样从 4 降到 3；
3. 最大迭代从 60 降到 40；
4. 风险预检时间点从 17 降到 13；
5. 保留 5000 点 dense verifier，不放松 `d_accept/d_stop`；
6. 复用确定性 Mesh 缓存，分离一次性加载耗时和在线优化耗时。

这属于 coarse-to-dense 预算分配，不是通过降低最终几何精度“刷实时性”。

## 3. 实验

对动态身体 A 和动态末端 C 各运行 15 次，共 30 次。每次执行完整优化和 dense verifier，分别记录 solver 收敛率、严格接受率和 dense 物理可行率。达到固定迭代上限不等于收敛；但若数值有限且独立 dense verifier 的目标、距离、限位和连续性检查全部通过，可作为“预算内物理可行候选”单独统计。控制耗时读取 P4 统一闭环逐场景统计并取最坏 p95。

## 4. 验收与回退

```text
planner p95 < 1000 ms
timeout_rate <= 0.05
dense candidate accept_rate >= 0.90
control p95 < 20 ms
```

若候选接受率下降，则优先恢复风险采样数，再增加迭代；不得降低 dense 安全距离阈值。

## 5. 运行

```bash
bash scripts/setup_ccro_p5.sh
```

输出 `data/results/ccro_p5/metrics.json`、`runs.jsonl`、`table_p5.md`。该耗时是本机 py310 仿真参考，不等同于真实控制器硬实时保证。

## 6. 实际执行结果（2026-06-23）

```text
runs                         30
mean                         848.65 ms
p95                          997.52 ms
max                          1006.35 ms
>1000 ms rate                0.0333
dense physical feasible      1.0000
strict solver acceptance     0.5000
solver convergence           0.5000
control worst p95            8.64 ms
```

首要 `1 Hz` 目标以 p95 口径刚好通过，但余量很小；增强 `<500 ms` 未完成。C 场景在 40 次迭代上限停止但独立 dense 物理检查通过，必须在论文中同时展示严格收敛率，不能只展示可行率。
