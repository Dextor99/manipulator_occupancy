#!/usr/bin/env python3
"""
纯机械臂 Y 轴往返运动测试 — 无感知 / 无安全逻辑 / 无可视化。

用途
----
验证机械臂在无外界干扰下：
  ① 末端沿 Y 轴走直线（而非弧线） — 使用 robotServiceLineMove
  ② 连续运动无停顿（累积相位积分，speed_scale 固定 1.0）
  ③ 正弦轨迹端点自然减速但不停死

用法
----
  python robot/test_y_oscillate.py                    # 默认 IP
  python robot/test_y_oscillate.py --ip 192.168.1.100
  python robot/test_y_oscillate.py --range 0.20       # ±20cm
  python robot/test_y_oscillate.py --omega 1.5         # 更快
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

# 确保项目根在 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot.robot_commander import RobotCommander


def main():
    p = argparse.ArgumentParser(description="纯 Y 轴往返运动测试")
    p.add_argument("--ip", default="192.168.123.96")
    p.add_argument("--range", type=float, default=0.40, help="Y 轴幅度 (m)")
    p.add_argument("--omega", type=float, default=0.8, help="角速度 (rad/s)")
    p.add_argument("--duration", type=float, default=60, help="运行时长 (s)")
    args = p.parse_args()

    print("=" * 60)
    print("纯机械臂 Y 轴往返运动测试")
    print(f"  IP:       {args.ip}")
    print(f"  Range:    ±{args.range}m")
    print(f"  Omega:    {args.omega} rad/s")
    print(f"  Duration: {args.duration}s")
    print(f"  Motion:   robotServiceLineMove (笛卡尔直线)")
    print("=" * 60)

    commander = RobotCommander(ip=args.ip, base_speed=0.05)
    if not commander.connect(home_joints_deg=[0.0, 0.0, 90.0, 0.0, 90.0, 0.0]):
        print("[错误] 连接失败")
        sys.exit(1)

    # 启动运动子进程
    # speed_scale 固定 1.0，不调速
    commander.start_y_oscillate(range_m=args.range)

    t0 = time.perf_counter()
    last_print = 0
    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed > args.duration:
                break

            y_pos = commander.get_y_pos()
            speed = commander.speed_scale

            # 每秒打印一次
            now = time.perf_counter()
            if now - last_print >= 1.0:
                dir_char = "→" if y_pos >= 0 else "←"
                print(f"  [{elapsed:5.1f}s]  Y={y_pos:+.4f}m {dir_char}  speed={speed:.0%}")
                last_print = now

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        commander.stop()
        print("测试结束")


if __name__ == "__main__":
    main()
