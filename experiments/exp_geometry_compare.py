import time

from camera.mock_reader import MockRGBDReader
from perception.geometry_fit import fit_aabb, fit_obb, fit_sphere


def timed(name, func, points):
    start = time.perf_counter()
    shape = func(points)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"{name}: radius={shape.radius:.4f} elapsed_ms={elapsed_ms:.3f}")


def main():
    points = MockRGBDReader().read().points_cam
    timed("sphere", fit_sphere, points)
    timed("aabb", fit_aabb, points)
    timed("obb", fit_obb, points)


if __name__ == "__main__":
    main()
