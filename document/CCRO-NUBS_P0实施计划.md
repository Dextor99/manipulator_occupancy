# CCRO-NUBS P0：版本、环境与实验基线冻结实施计划

> 目标：在不擅自提交用户工作树的前提下，用内容哈希、环境锁定、自动测试和阶段结果审计建立可追溯基线，为 P1–P7 提供统一起点。

---

## 1. 当前约束

当前 Git 工作树包含阶段 1–4 的大量未跟踪文件，且 `NUBSTrajectory-main/` 被 `.gitignore` 忽略。P0 不执行 `git add`、`git commit` 或历史重写，避免把用户尚未确认的文件直接提交。

冻结策略：

```text
Git HEAD
+ 工作树状态
+ 关键源码逐文件 SHA-256
+ 合并源码摘要
+ Python/C++/CMake 版本
+ 阶段 1–4 metrics.json 验收状态
+ planning 自动测试结果
```

这能够判断后续实验使用的代码是否与当前基线一致，即使当前文件尚未正式提交。

---

## 2. 新增文件

| 文件 | 作用 |
|---|---|
| `config/ccro_p0.yaml` | P0 输入、冻结范围和验收条件 |
| `config/ccro_environment_lock.yaml` | 已验证 py310 环境版本 |
| `experiments/exp_ccro_p0.py` | 环境、源码、测试和结果统一审计入口 |
| `scripts/setup_ccro_p0.sh` | 安装环境、构建扩展、测试并生成基线 |
| `.github/workflows/ccro-planning.yml` | 阶段 1–4 自动测试工作流 |
| `data/results/ccro_p0/` | manifest、metrics 和结果表 |

---

## 3. 冻结范围

```text
planning/
experiments/exp_ccro_stage*.py
experiments/exp_ccro_p0.py
config/ccro_stage*.yaml
config/ccro_p0.yaml
requirements-stage*.txt
requirements-ccro-lock.txt
scripts/build_ccro_stage1.sh
scripts/setup_ccro_*.sh
tests/planning/
NUBSTrajectory-main/include/
```

排除：

```text
__pycache__
build/
*.so
data/cache/
data/results/
```

---

## 4. 环境基线

```text
Python       3.10.15
NumPy        2.2.6
SciPy        1.15.3
PyYAML       6.0.2
Matplotlib   3.10.3
pybind11     3.0.4
pytest       9.1.1
Open3D       0.19.0
CMake        4.3.2
C++ compiler GCC 9.4.0
```

环境锁文件记录“当前已验证版本”，不是无限期禁止升级。升级任何核心依赖后必须重新执行 P0–P4 回归并生成新 manifest。

---

## 5. 自动验收

P0 通过条件：

1. 阶段 1–4 `metrics.json` 均存在且 `accepted=true`；
2. `tests/planning` 全部通过；
3. NUBS Python 扩展能够导入；
4. 所有配置和关键源码能够生成稳定 SHA-256；
5. Python 版本为 3.10，关键依赖可读取版本；
6. Git HEAD 可以读取。

Git 工作树不干净只记为警告，不自动判失败；正式论文归档前必须由用户确认并提交。

---

## 6. 运行方法

```bash
cd /home/hzy/Code/manipulator_occupancy
bash scripts/setup_ccro_p0.sh
```

只重新审计、不安装依赖：

```bash
/home/hzy/miniconda3/bin/conda run -n py310 \
  python -m experiments.exp_ccro_p0
```

输出：

```text
data/results/ccro_p0/
  config.yaml
  manifest.json
  metrics.json
  table_p0.md
```

---

## 7. P0 完成后进入 P1 的条件

```text
P0 accepted=true
planning tests 全部通过
阶段 1–4 结果全部可读取
源码摘要和环境摘要已保存
```

正式版本提交仍需用户确认；P1 的所有随机实验必须把 P0 的源码摘要写入结果文件。

---

## 8. 实际验收结果

```text
accepted                 true
planning tests           19 passed
阶段 1–4 accepted        全部 true
NUBS extension import    true
source files             以 manifest.json 为准
source SHA-256            ee866568d82ba395c28d2940c943e4166b014dc96cff15cac30e3856cbe9b27c
warnings                 1
```

警告为 Git 工作树存在未提交内容。P0 没有自动暂存或提交，而是用逐文件哈希冻结；P1 可以继续，正式论文归档前仍需人工确认版本提交。
