"""OpenCV-compatible export for celestial calibration results.

Converts a :class:`celestial_calibration.CalibrationResult` into the standard
OpenCV intrinsics tuple (camera matrix K, distortion coefficients) plus the
extrinsics (rvec, tvec) that drop directly into ``cv2.*`` functions. Writes
to four interchange formats:

  * **OpenCV YAML**   - readable by :func:`cv2.FileStorage` (``%YAML:1.0``)
  * **OpenCV JSON**   - same schema as cv2's JSON FileStorage output
  * **NumPy NPZ**     - readable by :func:`numpy.load`
  * **ROS camera_info YAML** - sensor_msgs/CameraInfo style

Distortion model
----------------
The native solver uses

    (x_d, y_d) = (1 + k1*r^2 + k2*r^4) * (x_n, y_n)

which is exactly the OpenCV pinhole + radial model with
``p1 = p2 = k3 = 0``. The exported ``dist_coeffs`` is therefore the canonical
5-vector ``[k1, k2, 0, 0, 0]`` and re-projecting a world direction with
``cv2.projectPoints`` produces the same pixel as our solver to floating-point
precision.

Extrinsics
----------
``rvec`` is the OpenCV-style Rodrigues vector of ``R_world_to_cam``; ``tvec``
is ``(0, 0, 0)`` because in our calibration the camera location is the origin
of the local ENU frame and celestial bodies are treated as unit-vector
directions at infinity.

Usage
-----
.. code-block:: python

    import cv2
    fs = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
    K    = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    undistorted = cv2.undistort(img, K, dist)
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import numpy as np

from celestial_calibration import (
    CalibrationResult,
    rotation_world_to_cam,
)


# ---------------------------------------------------------------------------
# Rodrigues rotation: R <-> r
# ---------------------------------------------------------------------------
def rodrigues_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a Rodrigues vector.

    The output ``rvec`` has direction equal to the rotation axis and magnitude
    equal to the rotation angle (in radians). Matches the convention used by
    :func:`cv2.Rodrigues`.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {R.shape}")
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, float(cos_theta)))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if abs(theta - math.pi) < 1e-6:
        # Near pi: extract axis from the symmetric part
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
        i = int(np.argmax(diag))
        v = np.zeros(3)
        v[i] = math.sqrt(max(0.0, (R[i, i] + 1.0) / 2.0))
        for j in range(3):
            if j != i:
                v[j] = R[i, j] / (2.0 * v[i]) if v[i] > 1e-12 else 0.0
        v *= theta
        return v
    factor = theta / (2.0 * math.sin(theta))
    return factor * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)


def matrix_from_rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rodrigues_from_matrix`. Returns a 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rvec / theta
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0],
    ], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------
# Calibration result -> OpenCV intrinsics tuple
# ---------------------------------------------------------------------------
def to_opencv_intrinsics(result: CalibrationResult) -> Dict[str, Any]:
    """Return a dict with everything OpenCV needs to use the calibration.

    Keys
    ----
    image_size       : (width, height) - tuple of ints
    camera_matrix    : 3x3 float64 K
    dist_coeffs      : (5,) float64 ``[k1, k2, 0, 0, 0]``
    rvec             : (3,) float64 Rodrigues vector of R_world_to_cam
    tvec             : (3,) float64 zeros
    rotation_matrix  : 3x3 float64 R_world_to_cam (same matrix, materialised)
    fov_horizontal_deg, fov_vertical_deg
    rms_reprojection_error_px, n_observations
    """
    f  = float(result.focal_length_px)
    cx = float(result.cx)
    cy = float(result.cy)
    K = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.array(
        [result.k1, result.k2, 0.0, 0.0, 0.0], dtype=np.float64
    )
    R = rotation_world_to_cam(
        math.radians(result.yaw_deg),
        math.radians(result.pitch_deg),
        math.radians(result.roll_deg),
    )
    rvec = rodrigues_from_matrix(R)
    tvec = np.zeros(3, dtype=np.float64)
    return {
        "image_size":       (int(result.image_width), int(result.image_height)),
        "camera_matrix":    K,
        "dist_coeffs":      dist,
        "rvec":             rvec,
        "tvec":             tvec,
        "rotation_matrix":  R,
        "fov_horizontal_deg":          float(result.fov_horizontal_deg),
        "fov_vertical_deg":            float(result.fov_vertical_deg),
        "rms_reprojection_error_px":   float(result.rms_reprojection_error_px),
        "n_observations":              int(result.n_observations),
    }


# ---------------------------------------------------------------------------
# OpenCV YAML / XML / JSON writers
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    """Canonical float formatting used inside OpenCV-compatible files."""
    return f"{v:.10g}"


def _opencv_matrix_yaml(name: str, mat: np.ndarray) -> str:
    """Render a NumPy matrix as a single OpenCV ``opencv-matrix`` block."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        rows, cols = 1, mat.shape[0]
    else:
        rows, cols = mat.shape
    flat = mat.reshape(-1)
    data_str = ", ".join(_fmt(v) for v in flat)
    return (
        f"{name}: !!opencv-matrix\n"
        f"   rows: {rows}\n"
        f"   cols: {cols}\n"
        f"   dt: d\n"
        f"   data: [ {data_str} ]"
    )


def export_opencv_yaml(result: CalibrationResult, path: str) -> str:
    """Write an OpenCV-style ``%YAML:1.0`` calibration file.

    Loadable directly by :class:`cv2.FileStorage`.
    """
    cv = to_opencv_intrinsics(result)
    W, H = cv["image_size"]
    now_local = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    parts = [
        "%YAML:1.0",
        "---",
        f'calibration_time: "{now_local}"',
        f"image_width: {W}",
        f"image_height: {H}",
        "flags: 0",
        'distortion_model: "pinhole_radial_k1k2"',
        'source: "celestial_calibration"',
        _opencv_matrix_yaml("camera_matrix",          cv["camera_matrix"]),
        _opencv_matrix_yaml("distortion_coefficients", cv["dist_coeffs"].reshape(1, -1)),
        _opencv_matrix_yaml("rotation_vector",        cv["rvec"].reshape(3, 1)),
        _opencv_matrix_yaml("translation_vector",     cv["tvec"].reshape(3, 1)),
        _opencv_matrix_yaml("rotation_matrix",        cv["rotation_matrix"]),
        f"avg_reprojection_error: {_fmt(cv['rms_reprojection_error_px'])}",
        f"n_observations: {cv['n_observations']}",
        f"fov_horizontal_deg: {_fmt(cv['fov_horizontal_deg'])}",
        f"fov_vertical_deg: {_fmt(cv['fov_vertical_deg'])}",
    ]
    text = "\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def export_opencv_json(result: CalibrationResult, path: str) -> str:
    """Write a JSON file mirroring cv2's ``FileStorage`` JSON layout."""
    cv = to_opencv_intrinsics(result)
    W, H = cv["image_size"]
    payload = {
        "image_width": W,
        "image_height": H,
        "distortion_model": "pinhole_radial_k1k2",
        "source": "celestial_calibration",
        "calibration_time_utc": datetime.now(timezone.utc).isoformat(),
        "camera_matrix":            _np_to_opencv_block(cv["camera_matrix"]),
        "distortion_coefficients":  _np_to_opencv_block(cv["dist_coeffs"].reshape(1, -1)),
        "rotation_vector":          _np_to_opencv_block(cv["rvec"].reshape(3, 1)),
        "translation_vector":       _np_to_opencv_block(cv["tvec"].reshape(3, 1)),
        "rotation_matrix":          _np_to_opencv_block(cv["rotation_matrix"]),
        "avg_reprojection_error":   cv["rms_reprojection_error_px"],
        "n_observations":           cv["n_observations"],
        "fov_horizontal_deg":       cv["fov_horizontal_deg"],
        "fov_vertical_deg":         cv["fov_vertical_deg"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _np_to_opencv_block(mat: np.ndarray) -> Dict[str, Any]:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        rows, cols = 1, mat.shape[0]
    else:
        rows, cols = mat.shape
    return {"rows": int(rows), "cols": int(cols), "dt": "d",
            "data": [float(v) for v in mat.reshape(-1)]}


def export_numpy_npz(result: CalibrationResult, path: str) -> str:
    """Write a ``.npz`` archive that can be loaded with :func:`numpy.load`.

    Provides ``camera_matrix``, ``dist_coeffs``, ``image_size``, ``rvec``,
    ``tvec``, and ``rotation_matrix`` arrays.
    """
    cv = to_opencv_intrinsics(result)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(
        path,
        camera_matrix=cv["camera_matrix"],
        dist_coeffs=cv["dist_coeffs"],
        image_size=np.array(cv["image_size"], dtype=np.int64),
        rvec=cv["rvec"],
        tvec=cv["tvec"],
        rotation_matrix=cv["rotation_matrix"],
        rms_reprojection_error_px=np.array([cv["rms_reprojection_error_px"]],
                                            dtype=np.float64),
    )
    if not path.endswith(".npz"):
        path = path + ".npz"
    return path


# ---------------------------------------------------------------------------
# ROS sensor_msgs/CameraInfo YAML
# ---------------------------------------------------------------------------
def export_ros_camera_info(
    result: CalibrationResult,
    path: str,
    camera_name: str = "celestial_camera",
) -> str:
    """Write a ROS ``camera_calibration`` YAML (sensor_msgs/CameraInfo).

    Uses the ``plumb_bob`` distortion model (5-element radial-tangential),
    populated with our ``[k1, k2, 0, 0, 0]``.
    """
    cv = to_opencv_intrinsics(result)
    W, H = cv["image_size"]
    K = cv["camera_matrix"]
    dist = cv["dist_coeffs"]
    P = np.zeros((3, 4), dtype=np.float64)
    P[:, :3] = K
    parts = [
        f"image_width: {W}",
        f"image_height: {H}",
        f"camera_name: {camera_name}",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        f"  data: [{', '.join(_fmt(v) for v in K.reshape(-1))}]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        "  cols: 5",
        f"  data: [{', '.join(_fmt(v) for v in dist)}]",
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        f"  data: [{', '.join(_fmt(v) for v in P.reshape(-1))}]",
    ]
    text = "\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# Convenience: project a celestial direction with the OpenCV-style model
# (used by the test suite to verify equivalence with the native model).
# ---------------------------------------------------------------------------
def project_with_opencv_model(
    direction_world: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> Tuple[float, float]:
    """Project a 3D world point/direction using the OpenCV pinhole+distortion
    pipeline. Equivalent to ``cv2.projectPoints`` but implemented in pure
    NumPy so the test suite does not depend on the cv2 binary.
    """
    direction_world = np.asarray(direction_world, dtype=np.float64).reshape(3)
    R = matrix_from_rodrigues(rvec)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    P_cam = R @ direction_world + tvec
    if P_cam[2] <= 1e-9:
        return float("nan"), float("nan")
    x = P_cam[0] / P_cam[2]
    y = P_cam[1] / P_cam[2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = (float(v) for v in dist[:5])
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    return fx * x_d + cx, fy * y_d + cy
