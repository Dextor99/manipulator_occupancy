# Chapter 6.3 Linux Static Benchmark Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan task by task. Check off each `- [ ]` item as it is completed.

## Goal

在 Linux + `py310` 环境中复用已经存在的 C++ `NUBSTrajectory` 实现，先编译并验证 `planning._nubs_cpp`，再运行第 6.3 节静态近场轨迹规划实验。实验代码放在 `experiments/new/6_3/`，正式结果放在 `data/results/6_3/`。

本计划不重写 NUBS 算法，不引入 Python 版替代实现，不把 MINCO 或 SciPy 轨迹作为 NUBS 替身。正确链路是：

```text
NUBSTrajectory.hpp
  -> planning/cpp/nubs_pybind.cpp
  -> planning/_nubs_cpp*.so
  -> planning/nubs_trajectory.py
  -> experiments/new/6_3/run_static_benchmark.py
```

## Scope Guard

- [ ] 不修改 6.3 的实验逻辑，除非现有实现与本计划的实验口径不一致。
- [ ] 不重写 `NUBSTrajectory6D` 的核心数学实现。
- [ ] 不添加 Python fallback 来绕过 `_nubs_cpp`。
- [ ] 不把 Windows 上的 `.pyd` 或旧 Linux `.so` 当作当前环境的有效产物。
- [ ] 先让 `_nubs_cpp` 在 Linux 当前 Python ABI 下导入成功，再运行 6.3 冒烟实验。

## Expected Linux Inputs And Outputs

输入：

- `planning/cpp/CMakeLists.txt`
- `planning/cpp/nubs_pybind.cpp`
- `planning/nubs_trajectory.py`
- `NUBSTrajectory/include/NUBSTrajectory.hpp` 或 `NUBSTrajectory-main/include/NUBSTrajectory.hpp`
- `config/ccro_stage2.yaml`

输出：

- `planning/_nubs_cpp*.so`
- `data/results/6_3_smoke/`
- `data/results/6_3/metrics.json`
- `data/results/6_3/frozen_instances/*.json`
- `data/results/6_3/trials/*.json`
- `data/results/6_3/paper/table_6_3_static_benchmark.md`
- `data/results/6_3/paper/figure_5_static_trajectory_and_jerk.png`

## Task 1: Verify Linux Environment And Source Layout

- [ ] 进入仓库根目录并确认分支状态。

```bash
cd /path/to/manipulator_occupancy-master
git status --short
```

- [ ] 激活或指定 `py310` 环境。

```bash
conda activate py310
python -VV
which python
```

- [ ] 确认依赖存在。

```bash
python - <<'PY'
import numpy, scipy, yaml, matplotlib, pybind11
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pybind11", pybind11.__version__)
print("pybind11 cmake", pybind11.get_cmake_dir())
PY

cmake --version
```

- [ ] 如缺少 CMake/Eigen/pybind11，则在 Linux 中安装。

```bash
conda install -n py310 -c conda-forge eigen=3.4 cmake ninja pybind11
```

- [ ] 确认已有 C++ NUBS 头文件。两个路径至少应有一个存在。

```bash
test -f NUBSTrajectory/include/NUBSTrajectory.hpp && echo "found NUBSTrajectory/include"
test -f NUBSTrajectory-main/include/NUBSTrajectory.hpp && echo "found NUBSTrajectory-main/include"
```

- [ ] 记录当前使用的 NUBS 头文件哈希，便于论文实验复现。

```bash
if test -f NUBSTrajectory/include/NUBSTrajectory.hpp; then
  sha256sum NUBSTrajectory/include/NUBSTrajectory.hpp
else
  sha256sum NUBSTrajectory-main/include/NUBSTrajectory.hpp
fi
```

Expected:

- `python -VV` 显示当前环境是 Python 3.10。
- `pybind11.get_cmake_dir()` 输出一个存在的 CMake 配置目录。
- 至少存在一个 `NUBSTrajectory.hpp`。

## Task 2: Make The Linux Build Script Robust

Modify: `scripts/build_ccro_stage1.sh`

- [ ] 将 build 目录改成 Linux ABI 专用目录，避免和 Windows 或旧构建缓存混用。

```bash
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/planning/cpp/build_py310_linux}"
```

- [ ] 自动检测 NUBS include 路径，优先使用当前本地目录 `NUBSTrajectory/include`，兼容旧目录 `NUBSTrajectory-main/include`。

```bash
if [[ -n "${NUBS_INCLUDE_DIR:-}" ]]; then
  :
elif [[ -f "${ROOT_DIR}/NUBSTrajectory/include/NUBSTrajectory.hpp" ]]; then
  NUBS_INCLUDE_DIR="${ROOT_DIR}/NUBSTrajectory/include"
elif [[ -f "${ROOT_DIR}/NUBSTrajectory-main/include/NUBSTrajectory.hpp" ]]; then
  NUBS_INCLUDE_DIR="${ROOT_DIR}/NUBSTrajectory-main/include"
else
  echo "NUBSTrajectory.hpp not found. Set NUBS_INCLUDE_DIR=/path/to/include" >&2
  exit 1
fi
```

- [ ] 保留现有 Conda 驱动方式，但打印关键路径。

```bash
echo "python prefix: ${PYTHON_PREFIX}"
echo "pybind11 dir: ${PYBIND11_DIR}"
echo "nubs include: ${NUBS_INCLUDE_DIR}"
echo "build dir: ${BUILD_DIR}"
```

- [ ] CMake 配置命令使用检测到的 include 路径。

```bash
"${CONDA_BIN}" run -n "${CONDA_ENV}" cmake \
  -S "${ROOT_DIR}/planning/cpp" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYTHON_PREFIX}" \
  -Dpybind11_DIR="${PYBIND11_DIR}" \
  -DNUBS_INCLUDE_DIR="${NUBS_INCLUDE_DIR}"
```

- [ ] 构建结束后立即验证导入。

```bash
"${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
from planning import _nubs_cpp
from planning.nubs_trajectory import NUBSTrajectory6D
print("built", _nubs_cpp.__file__)
print("wrapper", NUBSTrajectory6D.__name__)
PY
```

Expected:

- 脚本可以在 Linux 中生成 `planning/_nubs_cpp*.so`。
- 导入验证在脚本末尾直接通过。

## Task 3: Build And Verify `_nubs_cpp`

- [ ] 运行构建脚本。

```bash
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}" CONDA_ENV=py310 bash scripts/build_ccro_stage1.sh
```

- [ ] 确认 `.so` 是当前 Python ABI 的产物。

```bash
ls -lh planning/_nubs_cpp*.so
python - <<'PY'
from planning import _nubs_cpp
print(_nubs_cpp.__file__)
PY
```

- [ ] 运行 NUBS 单元测试。

```bash
python -m pytest tests/planning/test_nubs_stage1.py -q
```

- [ ] 运行 CCRO stage 2/3/4 相关测试，确认优化器链路没有被编译环境破坏。

```bash
python -m pytest \
  tests/planning/test_ccro_stage2.py \
  tests/planning/test_ccro_stage3.py \
  tests/planning/test_ccro_stage4.py \
  -q
```

Expected:

- `planning._nubs_cpp` 可导入。
- `tests/planning/test_nubs_stage1.py` 通过。
- stage 2/3/4 测试通过或只出现与本机性能阈值有关的可解释失败。

## Task 4: Verify 6.3 Code Before Running Experiments

- [ ] 确认 6.3 实验入口存在。

```bash
test -f experiments/new/6_3/run_static_benchmark.py
test -f experiments/new/6_3/plot_static_benchmark.py
```

- [ ] 运行 6.3 专用单元测试。

```bash
python -m pytest tests/test_ch6_3_new.py -q
```

- [ ] 语法检查实验入口。

```bash
python -m py_compile \
  experiments/new/6_3/run_static_benchmark.py \
  experiments/new/6_3/plot_static_benchmark.py
```

Expected:

- 单元测试通过。
- 语法检查通过。
- 如果此处失败，先修复 6.3 代码一致性问题，不进入实验运行。

## Task 5: Run One-Instance Smoke Benchmark

- [ ] 清楚记录这是冒烟目录，不是正式论文结果目录。

```bash
python -m experiments.new.6_3.run_static_benchmark \
  --config config/ccro_stage2.yaml \
  --output data/results/6_3_smoke \
  --instances-per-scenario 1 \
  --force-regenerate
```

- [ ] 检查冒烟输出完整性。

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/results/6_3_smoke")
assert (root / "metrics.json").is_file()
assert len(list((root / "frozen_instances").glob("*.json"))) == 3
assert len(list((root / "trials").glob("*.json"))) == 3
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
for scenario in ["A", "B", "C"]:
    assert scenario in metrics["scenarios"]
    for method in [
        "rrt_connect_smooth",
        "minco_risk",
        "nubs_without_risk",
        "critical_point_nubs",
        "ccro_nubs",
    ]:
        row = metrics["scenarios"][scenario]["methods"][method]
        assert "accepted_rate" in row
        assert "D_min" in row
        assert "J_smooth" in row
        assert "T_plan_ms" in row
print("smoke output ok")
PY
```

- [ ] 生成冒烟图。

```bash
python -m experiments.new.6_3.plot_static_benchmark \
  --input data/results/6_3_smoke \
  --output data/results/6_3_smoke/paper
```

Expected:

- `data/results/6_3_smoke/metrics.json` 存在。
- `data/results/6_3_smoke/frozen_instances/` 有 3 个实例。
- `data/results/6_3_smoke/trials/` 有 3 个 trial 文件。
- 冒烟图生成成功。

## Task 6: Run Full 30-Instance 6.3 Benchmark

- [ ] 只有 Task 5 冒烟通过后，才运行正式实验。

```bash
python -m experiments.new.6_3.run_static_benchmark \
  --config config/ccro_stage2.yaml \
  --output data/results/6_3 \
  --instances-per-scenario 10 \
  --force-regenerate
```

- [ ] 生成正式图表。

```bash
python -m experiments.new.6_3.plot_static_benchmark \
  --input data/results/6_3 \
  --output data/results/6_3/paper
```

- [ ] 审计正式输出数量和统计口径。

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/results/6_3")
assert (root / "metrics.json").is_file()
assert len(list((root / "frozen_instances").glob("*.json"))) == 30
assert len(list((root / "trials").glob("*.json"))) == 30
assert (root / "paper" / "table_6_3_static_benchmark.md").is_file()
assert (root / "paper" / "figure_5_static_trajectory_and_jerk.png").is_file()

metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
methods = [
    "rrt_connect_smooth",
    "minco_risk",
    "nubs_without_risk",
    "critical_point_nubs",
    "ccro_nubs",
]
for scenario in ["A", "B", "C"]:
    scenario_rows = metrics["scenarios"][scenario]
    for method in methods:
        row = scenario_rows["methods"][method]
        assert row["total_count"] == 10
        assert 0.0 <= row["accepted_rate"] <= 1.0
        assert "timeout_count" in row
print("full output ok")
PY
```

Expected:

- 30 个冻结实例，30 个 trial 文件。
- 每个场景 10 个实例。
- 表格包含 P1/P2/P3 x 5 方法共 15 行。
- `D_min` 和 `J_smooth` 只统计 dense verifier 接受的轨迹。
- `T_plan_ms` 统计所有尝试，超时或失败也计入耗时口径。

## Task 7: Refresh Chapter 6.8 Evidence Chain

Modify: `experiments/exp_68_summary.py`

- [ ] 确认 6.8 证据链引用新的 6.3 输出路径。

```python
"table": "data/results/6_3/paper/table_6_3_static_benchmark.md"
```

- [ ] 运行证据链汇总。

```bash
python -m experiments.exp_68_summary --output data/results/ch6_8
```

- [ ] 检查生成结果。

```bash
grep -n "6_3" data/results/ch6_8/table_6_8_evidence_chain.md
```

Expected:

- 6.8 证据链引用 `data/results/6_3/paper/table_6_3_static_benchmark.md`。
- 6.8 不再引用旧的 `ch6_3_sim` 或旧风险距离表作为 6.3 主实验结果。

## Task 8: Update Chapter 6.3 Manuscript Text

Modify: active manuscript file containing section 6.3.

- [ ] 找到当前论文第 6.3 节文件。

```bash
rg -n "6\\.3|CCRO-NUBS|RRT-Connect|MINCO" document result docs -S
```

- [ ] 将实验设计表述更新为：P1 末端近场、P2 中间连杆近场、P3 多障碍复杂近场，每类 10 个冻结扰动实例，共 30 个实例。
- [ ] 明确所有方法读取相同冻结实例，使用相同 AUBO i16 模型、起止构型、轨迹总时长、关节约束、采样间隔和 dense safety checker。
- [ ] 明确规划输入与安全验收分离：各方法使用带噪声/缺失的 observed points；统一 dense verifier 使用干净的 gt dense points。
- [ ] 主表只放五个主方法：`RRT-Connect + smoothing`、`MINCO-risk`、`NUBS without CCRO risk`、`Critical-point-NUBS`、`CCRO-NUBS`。
- [ ] 不把 CHOMP-inspired、TrajOpt-inspired、GPMP2-inspired 的轻量复现作为正文主外部基线。
- [ ] 指标口径写清楚：
  - `R_acc = N_accepted / 10`；
  - `D_min` 和 `J_smooth` 只对 accepted 轨迹报告均值和标准差；
  - `T_plan` 对全部实例统计；
  - 超时和 dense verifier 未通过都计入验收失败。
- [ ] 结论表述保持克制：CCRO-NUBS 的优势主要是全身风险一致性和轨迹平滑性，不声称其规划耗时绝对最优。

Expected:

- 6.3 正文叙述与 `data/results/6_3` 的实验口径一致。
- Figure 5 描述为空间轨迹/几何关系 + jerk norm 曲线。
- 不出现把 `fast-coarse` 写成实时在线主方法的表述。

## Failure Triage

### `_nubs_cpp` Import Fails

Run:

```bash
python - <<'PY'
import sys, sysconfig
print(sys.executable)
print(sys.version)
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY

ls -lh planning/_nubs_cpp*
ldd planning/_nubs_cpp*.so
```

Likely causes:

- `.so` 是旧 Python ABI 的产物。
- 当前 shell 没有使用 `py310`。
- `LD_LIBRARY_PATH` 或 Conda runtime 与构建环境不一致。

Resolution:

- 重新激活 `py310`。
- 使用 `planning/cpp/build_py310_linux` 重新配置和构建。
- 不复用 Windows `.pyd` 或旧 Linux `.so`。

### CMake Cannot Find Eigen3

Run:

```bash
conda install -n py310 -c conda-forge eigen=3.4
python - <<'PY'
import sys
print(sys.prefix)
PY
```

Then rebuild with:

```bash
CONDA_ENV=py310 bash scripts/build_ccro_stage1.sh
```

### CMake Cannot Find NUBSTrajectory.hpp

Run:

```bash
find . -path '*NUBSTrajectory.hpp' -print
```

Then rebuild with:

```bash
NUBS_INCLUDE_DIR=/absolute/path/to/NUBSTrajectory/include \
CONDA_ENV=py310 \
bash scripts/build_ccro_stage1.sh
```

### 6.3 Smoke Run Is Too Slow

- [ ] 停止正式 30 实例实验。
- [ ] 只保留 `--instances-per-scenario 1` 冒烟。
- [ ] 检查每个 trial JSON 中的方法级 `elapsed_ms`、`timeout`、`accepted`。
- [ ] 优先确认 RRT retries 是否按 3 个固定 seed 顺序运行并在第一个 dense-accepted 轨迹处停止。
- [ ] 确认优化器超时被记录为 `accepted=false`、`timeout=true`、`elapsed_ms=10000`，而不是无限等待。

## Definition Of Done

- [ ] Linux 中 `from planning import _nubs_cpp` 成功。
- [ ] `python -m pytest tests/planning/test_nubs_stage1.py -q` 通过。
- [ ] `python -m pytest tests/test_ch6_3_new.py -q` 通过。
- [ ] `data/results/6_3_smoke` 冒烟实验通过。
- [ ] `data/results/6_3` 正式 30 实例实验完成。
- [ ] `data/results/6_3/paper/table_6_3_static_benchmark.md` 和 `figure_5_static_trajectory_and_jerk.png` 生成。
- [ ] 6.8 证据链引用新的 6.3 结果。
- [ ] 论文第 6.3 节文字与新实验口径一致。

## Execution Order Summary

1. Linux 环境和 NUBS 头文件检查。
2. 优化 `scripts/build_ccro_stage1.sh`。
3. 编译 `_nubs_cpp*.so` 并运行 NUBS 测试。
4. 运行 6.3 单元测试。
5. 运行 `data/results/6_3_smoke` 冒烟。
6. 运行 `data/results/6_3` 正式实验。
7. 刷新 6.8 证据链和论文 6.3 正文。

## Self-Review

- 本计划把 Windows 编译问题从 6.3 实验逻辑中剥离出来，Linux 执行时先通过 `_nubs_cpp` 构建门槛。
- 本计划保留已有 C++ NUBS 实现，不建议重写 Python NUBS。
- 本计划使用 `experiments/new/6_3` 和 `data/results/6_3`，与用户指定目录一致。
- 本计划包含冒烟、完整实验、图表、证据链和论文文字的完整闭环。
