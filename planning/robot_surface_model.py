"""Deterministic multi-resolution URDF collision-mesh surface model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robot.urdf_model import URDFModel


def _numbers(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"expected three numbers, got {value!r}")
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    transform[:3, :3] = _rpy_matrix(_numbers(origin.get("rpy"), (0.0, 0.0, 0.0)))
    transform[:3, 3] = _numbers(origin.get("xyz"), (0.0, 0.0, 0.0))
    return transform


class RobotSurfaceModel:
    """Pre-sampled collision surfaces transformed by the existing URDF FK.

    The model accepts only the configured planning joints.  Other movable
    joints in the URDF (the gripper fingers in the current model) remain at
    their URDF zero value.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: list[str] | tuple[str, ...],
        density_totals: dict[str, int],
        *,
        seed: int = 20260623,
        min_points_per_link: int = 64,
        cache_dir: str | Path | None = "data/cache/robot_surface",
        geometry: str = "collision",
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.urdf = URDFModel(self.urdf_path)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 6 or len(set(self.joint_names)) != 6:
            raise ValueError("joint_names must contain six unique planning joints")
        unknown = set(self.joint_names) - set(self.urdf.joints)
        if unknown:
            raise ValueError(f"joint_names not found in URDF: {sorted(unknown)}")
        if geometry not in {"collision", "visual"}:
            raise ValueError("geometry must be 'collision' or 'visual'")
        required = {"coarse", "medium", "dense"}
        if set(density_totals) != required:
            raise ValueError(f"density_totals must have keys {sorted(required)}")
        self.density_totals = {name: int(value) for name, value in density_totals.items()}
        if not (
            0 < self.density_totals["coarse"]
            <= self.density_totals["medium"]
            <= self.density_totals["dense"]
        ):
            raise ValueError("density totals must be positive and coarse <= medium <= dense")
        self.seed = int(seed)
        self.min_points_per_link = max(int(min_points_per_link), 8)
        self.geometry = geometry
        self._mesh_specs = self._parse_mesh_specs()
        if not self._mesh_specs:
            raise RuntimeError(f"no {geometry} meshes found in {self.urdf_path}")
        self.link_names = tuple(spec["link"] for spec in self._mesh_specs)
        self._cache_key = self._make_cache_key()
        self._local: dict[str, dict[str, np.ndarray]] = {}
        cache_root = None if cache_dir is None else Path(cache_dir)
        if cache_root is not None and not cache_root.is_absolute():
            cache_root = self.urdf_path.parents[1] / cache_root
        self.cache_path = (
            None if cache_root is None else cache_root / f"surface_{self._cache_key}.npz"
        )
        if self.cache_path is not None and self.cache_path.exists():
            self._load_cache()
        else:
            self._sample_meshes()
            if self.cache_path is not None:
                self._save_cache()

    def _parse_mesh_specs(self) -> list[dict]:
        root = ET.parse(self.urdf_path).getroot()
        specs: list[dict] = []
        for link in root.findall("link"):
            geometry_node = link.find(self.geometry)
            if geometry_node is None:
                continue
            mesh = geometry_node.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            path = (self.urdf_path.parent / mesh.get("filename")).resolve()
            if not path.exists():
                continue
            specs.append(
                {
                    "link": link.get("name"),
                    "path": path,
                    "origin": _origin_transform(geometry_node.find("origin")),
                    "scale": _numbers(mesh.get("scale"), (1.0, 1.0, 1.0)),
                }
            )
        return specs

    def _make_cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.urdf_path.read_bytes())
        digest.update(self.geometry.encode())
        digest.update(str(self.seed).encode())
        digest.update(str(self.min_points_per_link).encode())
        digest.update(json.dumps(self.density_totals, sort_keys=True).encode())
        for spec in self._mesh_specs:
            digest.update(spec["link"].encode())
            digest.update(spec["path"].read_bytes())
            digest.update(np.asarray(spec["origin"]).tobytes())
            digest.update(np.asarray(spec["scale"]).tobytes())
        return digest.hexdigest()[:20]

    def _sample_meshes(self) -> None:
        import open3d as o3d

        meshes: list[tuple[dict, object, float]] = []
        for spec in self._mesh_specs:
            mesh = o3d.io.read_triangle_mesh(str(spec["path"]))
            if not mesh.has_triangles():
                raise RuntimeError(f"mesh has no triangles: {spec['path']}")
            vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
            vertices *= spec["scale"][None, :]
            if np.linalg.norm(np.ptp(vertices, axis=0)) > 10.0:
                vertices *= 0.001
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            area = float(mesh.get_surface_area())
            if not np.isfinite(area) or area <= 0.0:
                raise RuntimeError(f"invalid mesh surface area: {spec['path']}")
            meshes.append((spec, mesh, area))

        areas = np.asarray([item[2] for item in meshes], dtype=np.float64)
        dense_target = self.density_totals["dense"]
        dense_counts = np.maximum(
            self.min_points_per_link,
            np.rint(dense_target * areas / np.sum(areas)).astype(int),
        )
        for index, ((spec, mesh, _), dense_count) in enumerate(zip(meshes, dense_counts)):
            o3d.utility.random.seed(self.seed + index)
            cloud = mesh.sample_points_uniformly(number_of_points=int(dense_count))
            points = np.asarray(cloud.points, dtype=np.float64)
            origin = spec["origin"]
            dense = points @ origin[:3, :3].T + origin[:3, 3]
            levels: dict[str, np.ndarray] = {"dense": np.ascontiguousarray(dense)}
            for level in ("medium", "coarse"):
                ratio = self.density_totals[level] / self.density_totals["dense"]
                count = min(len(dense), max(16, int(round(len(dense) * ratio))))
                indices = np.linspace(0, len(dense) - 1, count).round().astype(np.int64)
                levels[level] = np.ascontiguousarray(dense[indices])
            self._local[spec["link"]] = levels

    def _save_cache(self) -> None:
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "metadata": np.asarray(
                json.dumps(
                    {
                        "cache_key": self._cache_key,
                        "links": list(self.link_names),
                        "geometry": self.geometry,
                    }
                )
            )
        }
        for index, link in enumerate(self.link_names):
            for level in ("coarse", "medium", "dense"):
                payload[f"link_{index}_{level}"] = self._local[link][level]
        np.savez_compressed(self.cache_path, **payload)

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        with np.load(self.cache_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("cache_key") != self._cache_key:
                raise RuntimeError("robot surface cache metadata mismatch")
            if tuple(metadata.get("links", [])) != self.link_names:
                raise RuntimeError("robot surface cache link order mismatch")
            for index, link in enumerate(self.link_names):
                self._local[link] = {
                    level: np.ascontiguousarray(data[f"link_{index}_{level}"], dtype=np.float64)
                    for level in ("coarse", "medium", "dense")
                }

    def _joint_dict(self, q: np.ndarray) -> dict[str, float]:
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("q must be a finite array with shape (6,)")
        return {name: float(values[index]) for index, name in enumerate(self.joint_names)}

    def local_samples(self, link: str, density: str = "medium") -> np.ndarray:
        if link not in self._local:
            raise KeyError(link)
        if density not in {"coarse", "medium", "dense"}:
            raise ValueError("density must be coarse, medium, or dense")
        return self._local[link][density].copy()

    def surface_by_link(
        self,
        q: np.ndarray,
        density: str = "medium",
        links: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        if density not in {"coarse", "medium", "dense"}:
            raise ValueError("density must be coarse, medium, or dense")
        fk = self.urdf.link_transforms(self._joint_dict(q))
        selected = set(self.link_names) if links is None else set(links)
        unknown = selected - set(self.link_names)
        if unknown:
            raise KeyError(f"unknown surface links: {sorted(unknown)}")
        result: dict[str, np.ndarray] = {}
        for link in self.link_names:
            if link not in selected:
                continue
            local = self._local[link][density]
            transform = fk.get(link)
            if transform is None:
                continue
            result[link] = local @ transform[:3, :3].T + transform[:3, 3]
        return result

    def surface(
        self,
        q: np.ndarray,
        density: str = "medium",
        links: set[str] | None = None,
    ) -> np.ndarray:
        by_link = self.surface_by_link(q, density=density, links=links)
        return np.vstack(list(by_link.values())) if by_link else np.empty((0, 3))

    def point_jacobian(self, q: np.ndarray, link: str, local_point: np.ndarray) -> np.ndarray:
        """Analytic translational Jacobian of a link-local surface point."""
        values = np.asarray(q, dtype=np.float64)
        local = np.asarray(local_point, dtype=np.float64)
        fk, frames = self.urdf.link_transforms_with_joint_frames(self._joint_dict(values))
        transform = fk[link]
        point_world = local @ transform[:3, :3].T + transform[:3, 3]
        chain = set(self.urdf.joint_chain_to_link(link))
        jacobian = np.zeros((3, len(self.joint_names)), dtype=np.float64)
        for col, joint_name in enumerate(self.joint_names):
            if joint_name not in chain:
                continue
            joint = self.urdf.joints[joint_name]
            origin, axis = frames[joint_name]
            if joint.type == "revolute":
                jacobian[:, col] = np.cross(axis, point_world - origin)
            elif joint.type == "prismatic":
                jacobian[:, col] = axis
        return jacobian

    def sample_counts(self) -> dict[str, dict[str, int]]:
        return {
            link: {level: len(points) for level, points in levels.items()}
            for link, levels in self._local.items()
        }
