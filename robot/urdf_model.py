"""Lightweight URDF parser + forward kinematics solver (pure numpy)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _xyz(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()])


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ roll-pitch-yaw → 3×3 rotation matrix."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr],
    ])


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """3×3 rotation matrix for a rotation of *angle* rad about *axis*."""
    a = axis.astype(float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a /= n
    x, y, z = a
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + x * x * (1 - c),      x * y * (1 - c) - z * s,  x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s,  c + y * y * (1 - c),      y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s,  z * y * (1 - c) + x * s,  c + z * z * (1 - c)],
    ])


class Joint:
    """A URDF joint with fixed (origin) and variable (angle / translation) parts."""

    def __init__(self, name: str, type_: str,
                 origin_xyz: np.ndarray, origin_rpy: np.ndarray,
                 parent: str, child: str,
                 axis: np.ndarray):
        self.name = name
        self.type = type_              # 'revolute' | 'prismatic' | 'fixed'
        self.parent = parent
        self.child = child
        self._fixed = _transform(origin_xyz, origin_rpy)
        self._axis = axis.copy()

    def variable(self, value: float) -> np.ndarray:
        """4×4 variable part of the joint transform given *value*."""
        if self.type == 'revolute':
            R = _rodrigues(self._axis, value)
            out = np.eye(4)
            out[:3, :3] = R
            return out
        if self.type == 'prismatic':
            out = np.eye(4)
            out[:3, 3] = self._axis * value
            return out
        return np.eye(4)  # fixed


class URDFModel:
    """Load a URDF and compute forward kinematics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.joints: dict[str, Joint] = {}
        self.links: dict[str, dict] = {}       # link_name → meta
        self.root_link: str = ''
        self._order: list[str] = []            # joint names, root → leaf
        self._parse()
        self._topological_order()

    # ── parsing ────────────────────────────────────────────────

    def _parse(self):
        tree = ET.parse(self.path)
        root = tree.getroot()

        for elem in root.iter('link'):
            name = elem.get('name')
            vis = elem.find('visual')
            mesh_rel = None
            vis_origin = np.eye(4)
            if vis is not None:
                o = vis.find('origin')
                if o is not None:
                    vis_origin = _transform(_xyz(o.get('xyz', '0 0 0')),
                                            _xyz(o.get('rpy', '0 0 0')))
                g = vis.find('geometry')
                if g is not None:
                    m = g.find('mesh')
                    if m is not None:
                        mesh_rel = m.get('filename')
            self.links[name] = dict(visual_mesh=mesh_rel,
                                    visual_origin=vis_origin)

        for elem in root.iter('joint'):
            name = elem.get('name')
            type_ = elem.get('type', 'fixed')
            o = elem.find('origin')
            xyz = _xyz(o.get('xyz', '0 0 0')) if o is not None else np.zeros(3)
            rpy = _xyz(o.get('rpy', '0 0 0')) if o is not None else np.zeros(3)
            parent = elem.find('parent').get('link')
            child = elem.find('child').get('link')
            a = elem.find('axis')
            axis = _xyz(a.get('xyz', '1 0 0')) if a is not None else np.array([1., 0., 0.])
            self.joints[name] = Joint(name, type_, xyz, rpy, parent, child, axis)

    def _topological_order(self):
        children = {j.child for j in self.joints.values()}
        parents = {j.parent for j in self.joints.values()}
        self.root_link = next(iter(parents - children), 'base_link')

        adj: dict[str, list[tuple[str, str]]] = {}
        for n, j in self.joints.items():
            adj.setdefault(j.parent, []).append((n, j.child))

        order = []
        seen = set()

        def dfs(link: str):
            if link in seen:
                return
            seen.add(link)
            for jname, child in adj.get(link, []):
                order.append(jname)
                dfs(child)

        dfs(self.root_link)
        self._order = order

    # ── public API ─────────────────────────────────────────────

    def movable_joints(self) -> list[str]:
        return [n for n, j in self.joints.items() if j.type != 'fixed']

    def link_transforms(self, angles: dict[str, float]) -> dict[str, np.ndarray]:
        """Return {link_name: 4×4 world→link} for every link in the tree."""
        T = {self.root_link: np.eye(4)}
        for jname in self._order:
            j = self.joints[jname]
            v = angles.get(jname, 0.0)
            T[j.child] = T[j.parent] @ j._fixed @ j.variable(v)
        return T

    _PREFERRED = ('.obj', '.STL', '.stl', '.ply')  # open3d-readable

    def resolve_mesh(self, link_name: str) -> str | None:
        """Absolute path to the link's visual mesh (prefers open3d-readable ext)."""
        info = self.links.get(link_name)
        if info is None or info['visual_mesh'] is None:
            return None
        rel = info['visual_mesh']
        path = self.path.parent / rel
        if path.suffix.lower() not in ('.dae',):
            return str(path) if path.exists() else None
        # Try preferred extensions before falling back to .dae
        for ext in self._PREFERRED:
            alt = path.with_suffix(ext)
            if alt.exists():
                return str(alt)
        return str(path) if path.exists() else None
