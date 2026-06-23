# CCRO-NUBS P3：真实动态序列真值增强与影子重规划实施计划

## 1. 目标与证据边界

P3 建立“录制数据审计—真值导入—预测评价—影子规划日志”闭环。已有 8 条动态接近和 3 条接近/停留/离开序列可直接审计；但原始录制未同步 AprilTag、导轨或人工三维框真值，因此不能把背景差分弱参考冒充中心真值，也不能虚构真实 RMSE。

本阶段把可自动完成部分全部完成，同时将真实中心真值状态明确标为 `pending_real_truth`。这不阻碍 P4–P6 仿真工作，但 P3 论文真值验收只有补录/标注后才能转为完成。

## 2. 新增程序和数据格式

- `experiments/p3_truth_io.py`：严格读取真值 CSV 并执行因果 CV 预测评价；
- `experiments/exp_ccro_p3.py`：审计 11 条录制、汇总已有预警结果、运行程序化真值基准、生成影子日志；
- `data/annotations/ccro_p3_truth_template.csv`：真值模板；
- `scripts/setup_ccro_p3.sh`：自动测试和运行。

真值列：

```text
sequence,timestamp,x,y,z,radius,source
```

时间戳必须与 RGB-D 使用同一时钟或给出已标定偏移。预测严格只使用当前及历史样本，未来真值只用于评分。

## 3. 自动实验

1. 检查 manifest、实际帧数、时间戳有限/单调和机器人关节维数；
2. 汇总已有 8 条动态、3 条恢复序列的预警结果；
3. 对匀速、停留、反向、加速四类带噪程序化真值计算 `RMSE/p95/覆盖率`；
4. 将阶段 4 每次候选的提交、耗时、接受/拒绝、安全收益写为统一影子日志；
5. 若真值 CSV 存在，自动增加真实 RMSE 和覆盖率并重新判定。

## 4. 验收

```text
recorded_sequences >= 10
有效录制序列 >= 10；同时报告并隔离时间戳异常目录
synthetic prediction RMSE <= 0.06 m
synthetic coverage >= 0.80
shadow log 字段完整
```

`simulation_accepted=true` 表示程序链通过；只有真实 CSV 存在且有效时才允许 `real_truth_complete=true`。

## 5. 后续实物/标注步骤

优先采用固定导轨或 AprilTag：标定 `base_T_tag_camera`，每帧保存原始检测时间戳和中心，不要只保存滤波结果。至少覆盖接近、停留、离开、突然停止、反向和变速，每类 3 条。运行：

```bash
bash scripts/setup_ccro_p3.sh
```

影子模式禁止向机械臂发送轨迹，只记录候选及假定切换结果。

## 6. 实际执行结果（2026-06-23）

```text
发现录制目录              14
有效时间戳目录            12
timestamp_valid_rate      0.8571
程序化预测 pooled RMSE    0.01708 m
程序化预测覆盖率          0.9809
标准化影子记录            3
simulation_accepted       true
real_truth_complete       false
```

`ch4_3_dynamic_01` 与 `ch4_3_recover_01` 存在时间戳倒退，已在 `recording_audit.json` 中隔离。真实真值 CSV 尚不存在，因此总 `accepted=false` 是有意的证据边界，不是程序失败。
