"""从标定结果文件夹读取 calibration.json，生成 camera_extrinsic.json。

直接在程序下方修改 CALIB_DIR 变量即可。
"""

import json
from pathlib import Path

import numpy as np

# ============================================================
# ★ 在这里指定标定结果文件夹（支持三种方式）:
#   1. 文件夹名:        "effector_real_202606020919"
#   2. 相对路径:        "robot/01_calibrate_robot/.../effector_real_202606020919"
#   3. 绝对路径:        "/home/.../effector_real_202606020919"
# ============================================================
CALIB_DIR = "effector_real_202606021516"
# ============================================================

# 标定记录根目录
RECORD_BASE = Path(__file__).resolve().parent.parent / "robot" / "01_calibrate_robot" / "python" / "_01_dataset_collect" / "record"

# 输出路径
OUTPUT_PATH = Path(__file__).resolve().parent / "camera_extrinsic.json"


def find_calibration_dir(folder: str) -> Path:
    """根据输入找到 calibration.json 所在目录。"""
    candidate = Path(folder)

    # 1. 绝对路径
    if candidate.is_absolute():
        if (candidate / "calibration.json").is_file():
            return candidate
        raise FileNotFoundError(f"未在 {candidate} 中找到 calibration.json")

    # 2. 相对路径，直接存在
    if (candidate / "calibration.json").is_file():
        return candidate.resolve()

    # 3. 在 RECORD_BASE 下查找
    candidate = RECORD_BASE / folder
    if (candidate / "calibration.json").is_file():
        return candidate

    # 4. 模糊前缀匹配
    matches = sorted(
        [d for d in RECORD_BASE.iterdir() if d.is_dir() and d.name.startswith(folder)],
        reverse=True,
    )
    if matches:
        print(f"模糊匹配到: {matches[0].name}")
        return matches[0]

    raise FileNotFoundError(
        f"未找到标定文件夹 '{folder}'\n"
        f"  查找路径: {RECORD_BASE}\n"
        f"  可用文件夹: {[d.name for d in RECORD_BASE.iterdir() if d.is_dir() and d.name.startswith('effector_real_')]}"
    )


def load_calibration(calib_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载 calibration.json，返回 (R, t)。优先使用优化后的 opt_ 结果。"""
    path = calib_dir / "calibration.json"
    with open(path, "r") as f:
        data = json.load(f)

    if "opt_Rcam2base" in data and "opt_tcam2base" in data:
        R = np.array(data["opt_Rcam2base"], dtype=float)
        t = np.array(data["opt_tcam2base"], dtype=float)
        print("  使用优化后标定结果 (opt_Rcam2base / opt_tcam2base)")
    elif "Rcam2base" in data and "tcam2base" in data:
        R = np.array(data["Rcam2base"], dtype=float)
        t = np.array(data["tcam2base"], dtype=float)
        print("  使用原始标定结果 (Rcam2base / tcam2base)")
    else:
        raise ValueError(f"calibration.json 中未找到标定矩阵: {path}")

    return R, t


def build_extrinsic(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """组装 4×4 变换矩阵 base_T_cam。"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()[:3]
    return T


def save_extrinsic(T: np.ndarray, output_path: Path) -> None:
    """保存为 camera_extrinsic.json。"""
    data = {"base_T_cam": T.tolist()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  已写入: {output_path}")


def main():
    try:
        calib_dir = find_calibration_dir(CALIB_DIR)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    print(f"标定文件夹: {calib_dir}")
    R, t = load_calibration(calib_dir)
    print(f"  旋转矩阵 R:\n{R}")
    print(f"  平移向量 t: {t.flatten()}")

    T = build_extrinsic(R, t)
    print(f"  组装 base_T_cam:\n{T}")

    save_extrinsic(T, OUTPUT_PATH)
    print("完成!")


if __name__ == "__main__":
    main()
