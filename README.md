# Manipulator Occupancy

Non-ROS Python prototype for unified spatio-temporal occupancy perception and lightweight risk modeling around a manipulator.

The default path is hardware-free:

```bash
pip install -r requirements.txt
python main.py --source mock --max-frames 100
pytest -q
```

The mock pipeline runs:

```text
mock RGB-D/point cloud
-> camera-to-base transform
-> workspace crop and voxel downsample
-> robot capsule self-filter
-> external clustering
-> sphere/AABB/OBB fitting
-> temporal tracking and velocity estimation
-> short-horizon risk prediction
-> capsule-sphere distance check
-> SAFE/WARNING/SLOW/STOP decision
```

## Hardware Notes

- `config/camera_extrinsic.json` is an identity transform placeholder. Replace it with a calibrated `base_T_cam` before real experiments.
- `config/robot_model.yaml` points to an optional URDF. If the URDF path is missing, the system uses a simple mock robot model.
- `camera/realsense_reader.py` raises a clear error if `pyrealsense2` is not installed.
- `control/robot_command.py` only prints mock commands in v1. Add your manipulator SDK adapter behind the same interface.

## Main Commands

```bash
python main.py --source mock --max-frames 100
python main.py --source mock --visualize --max-frames 300
python experiments/exp_self_filter.py
python experiments/exp_geometry_compare.py
python experiments/exp_prediction.py
python experiments/exp_safety_motion.py
```
