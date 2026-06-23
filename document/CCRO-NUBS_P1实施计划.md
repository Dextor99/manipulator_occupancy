# CCRO-NUBS P1：多随机种子动态鲁棒性统计实施计划

> 目标：在阶段 3 名义 A/B/C dense 验收基础上，对障碍中心、速度和半径施加可重复扰动，统计当前帧、仅末端时空和全身时空方法的成功率与身体漏检率。

---

## 1. 实验分层

```text
名义正确性：阶段 3 dense Mesh + 0.025 s，已通过
统计鲁棒性：P1 medium Mesh + 0.05 s，30 个随机 trial
```

P1 不用 medium 结果替代 dense 最终验收。其作用是扩大随机样本并筛查方法趋势；对失败或临界 trial 后续应再执行 dense 复核。

---

## 2. 扰动模型

每个对象独立采样：

```text
center noise    N(0, 0.003^2) m
velocity noise  N(0, 0.005^2) m/s
radius noise    N(0, 0.002^2) m
```

随机种子从 `20260701` 开始，共 30 次。所有方法在同一 trial 使用完全相同的扰动占据。

---

## 3. 对比方法

```text
baseline       无风险 NUBS
current_full   只使用当前帧优化所得轨迹
temporal_ee    只考虑末端/夹爪时空风险
temporal_full  URDF/Mesh 全身时空风险
```

轨迹采用阶段 3 已保存的优化结果，不在每个扰动 trial 重新调参，从而评价名义轨迹对小预测误差的鲁棒性。

---

## 4. 指标与验收

```text
pass_rate                D_min >= d_stop 的比例
D_min mean/p05/min
time_below_d_stop mean
current_frame_failure    当前帧方法失败比例
body_counterexample      B 中 EE 失败而 full 通过比例
```

验收门槛：

```text
temporal_full 总体 pass_rate >= 0.90
baseline 总体 pass_rate <= 0.20
current_full 失败比例 >= 0.80
B 身体反例比例 >= 0.80
阶段 3 名义 dense accepted=true
```

---

## 5. 运行方法

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_p1
```

输出：

```text
data/results/ccro_p1/
  config.yaml
  metrics.json
  trials.jsonl
  table_p1.md
```

---

## 6. 实际执行结果（2026-06-23）

自动实验完成 `30 trial/scenario`、共 `360` 次方法—场景评价，最终 `accepted=true`：

```text
baseline pooled pass_rate       0.0000
current_full pooled pass_rate   0.0000
temporal_ee pooled pass_rate    0.3444
temporal_full pooled pass_rate  0.9778
B 身体反例复现率                1.0000
temporal_full C pass_rate       0.9333
```

这说明全身时空方法对设定的小幅中心、速度和半径扰动具有稳定趋势；C 场景仍有 2 个失败 trial，已保留在 `trials.jsonl`，后续必须做 dense 复核而不是删除异常样本。
