"""Math helpers shared by env / rewards.

PyBullet quaternion convention is (x, y, z, w). Numpy arrays are float64.
"""
from __future__ import annotations

import numpy as np
import pybullet as p


def quat_to_axisangle3(q) -> np.ndarray:
    """Quaternion (xyzw) → axis*angle vector (rad).

    Identity → zeros. ‖result‖ ∈ [0, π]. Used to express orientation error
    in 3 dims (spec §2.3) instead of the 4-dim quaternion.
    """
    q = np.asarray(q, dtype=np.float64)
    if q[3] < 0:
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = float(np.sqrt(max(1.0 - w * w, 1e-12)))
    if s < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return (q[:3] / s) * angle


def quat_diff(q_from, q_to) -> np.ndarray:
    """Relative quaternion sending frame `from` to frame `to`."""
    return np.asarray(p.getDifferenceQuaternion(list(q_from), list(q_to)),
                      dtype=np.float64)


def relative_axisangle(q_from, q_to) -> np.ndarray:
    """Orientation error from `q_from` to `q_to` as a 3-vec axis*angle."""
    return quat_to_axisangle3(quat_diff(q_from, q_to))


def quat_axis(q, axis: str = "z") -> np.ndarray:
    """World-frame direction of a local axis under rotation q."""
    R = np.array(p.getMatrixFromQuaternion(list(q))).reshape(3, 3)
    return R[:, "xyz".index(axis)].astype(np.float64)


def angle_between(v1, v2) -> float:
    """Unsigned angle (rad) between two 3-vectors."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
    return float(np.arccos(c))


def rpy_to_quat(rpy) -> np.ndarray:
    return np.asarray(p.getQuaternionFromEuler(list(rpy)), dtype=np.float64)


def quat_multiply(q1, q2) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both xyzw."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def axisangle3_to_quat(v) -> np.ndarray:
    """Inverse of quat_to_axisangle3. v = axis * angle (rad), 3-vec → xyzw."""
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = v / angle
    half = angle * 0.5
    s = float(np.sin(half))
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, float(np.cos(half))],
                    dtype=np.float64)
