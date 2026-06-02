from camera.mock_reader import MockRGBDReader
from perception.self_filter import filter_robot_self_points
from robot.capsule_model import mock_capsules


def main():
    frame = MockRGBDReader().read()
    external, robot = filter_robot_self_points(frame.points_cam, mock_capsules(), margin=0.03)
    print(f"raw_points={len(frame.points_cam)} robot_points={len(robot)} external_points={len(external)}")


if __name__ == "__main__":
    main()
