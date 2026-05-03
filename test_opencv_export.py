"""Self-tests for opencv_export.py.

Run inside the project conda environment:

    conda activate cg
    python test_opencv_export.py

The tests verify:
  - Rodrigues forward + inverse round-trips to ~machine precision
  - to_opencv_intrinsics produces a well-formed K, dist, rvec, tvec
  - The exported (K, dist, rvec) re-projects celestial directions to the
    same pixels as our native solver (to floating-point precision).
  - YAML / JSON / NPZ files contain the expected matrices
  - Optional cv2 round-trip if cv2 is installed (sanity-check vs OpenCV itself)
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import sys
from datetime import datetime, timezone

import numpy as np

import celestial_calibration as cc
import opencv_export as ox


# ---------------------------------------------------------------------------
# Synthetic CalibrationResult for tests
# ---------------------------------------------------------------------------
def _make_synth_result():
    """Build a CalibrationResult with values small enough to keep tests fast."""
    truth = dict(f=950.0, yaw_deg=130.0, pitch_deg=10.0, roll_deg=-1.5,
                 k1=-0.07, k2=0.015)
    W, H = 1280, 720
    cx, cy = W/2.0, H/2.0
    az_el = []
    pixels = []
    bodies = []
    rng = np.random.default_rng(0)
    base_az, base_el = truth["yaw_deg"], truth["pitch_deg"]
    for i in range(8):
        az = base_az + rng.uniform(-25, 25)
        el = base_el + rng.uniform(-12, 12)
        u, v = cc.project_az_el(
            az, el, truth["f"],
            math.radians(truth["yaw_deg"]),
            math.radians(truth["pitch_deg"]),
            math.radians(truth["roll_deg"]),
            truth["k1"], truth["k2"], cx, cy,
        )
        az_el.append((az, el))
        pixels.append((u, v))
        bodies.append(f"body_{i}")
    return cc.CalibrationResult(
        focal_length_px=truth["f"],
        yaw_deg=truth["yaw_deg"], pitch_deg=truth["pitch_deg"],
        roll_deg=truth["roll_deg"],
        k1=truth["k1"], k2=truth["k2"],
        image_width=W, image_height=H, cx=cx, cy=cy,
        rms_reprojection_error_px=0.0, max_reprojection_error_px=0.0,
        median_reprojection_error_px=0.0, n_observations=len(pixels),
        converged=True, cost=0.0,
        fov_horizontal_deg=2*math.degrees(math.atan(W/2.0/truth["f"])),
        fov_vertical_deg=2*math.degrees(math.atan(H/2.0/truth["f"])),
        per_obs_residuals_px=[(0.0, 0.0)] * len(pixels),
        per_obs_predicted_px=list(pixels),
        per_obs_observed_px=list(pixels),
        per_obs_az_el_deg=list(az_el),
        per_obs_body=bodies,
    ), truth, pixels, az_el


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_rodrigues_roundtrip():
    """R -> rvec -> R must round-trip to ~machine precision for all
    representative rotation magnitudes."""
    rng = np.random.default_rng(1)
    cases = [
        np.array([0.0, 0.0, 0.0]),                           # identity
        np.array([0.5, 0.3, -0.2]),                          # generic
        np.array([1.5, 0.0, 0.0]),                           # near pi/2 about x
        np.array([0.0, math.pi - 1e-3, 0.0]),                # near pi
        np.array([math.pi/3, math.pi/3, math.pi/3]) / math.sqrt(3) * 2.5,
    ]
    for _ in range(20):
        # Random rotation axis and angle in [0, pi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0.05, math.pi - 0.05)
        cases.append(axis * ang)

    max_err = 0.0
    for r in cases:
        R = ox.matrix_from_rodrigues(r)
        # R must be a proper rotation
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        r2 = ox.rodrigues_from_matrix(R)
        R2 = ox.matrix_from_rodrigues(r2)
        err = float(np.max(np.abs(R - R2)))
        max_err = max(max_err, err)
        assert err < 1e-9, f"rodrigues round-trip failed: r={r}, err={err}"
    print(f"[OK] rodrigues round-trip ({len(cases)} cases) max |dR| = {max_err:.2e}")


def test_to_opencv_intrinsics_shapes_and_values():
    res, truth, _, _ = _make_synth_result()
    cv = ox.to_opencv_intrinsics(res)
    assert cv["image_size"] == (1280, 720)
    K = cv["camera_matrix"]
    assert K.shape == (3, 3) and K.dtype == np.float64
    assert K[0, 0] == K[1, 1] == truth["f"], f"fx/fy mismatch {K}"
    assert K[0, 2] == 640.0 and K[1, 2] == 360.0
    assert K[2, 2] == 1.0 and K[0, 1] == 0.0 and K[1, 0] == 0.0
    dist = cv["dist_coeffs"]
    assert dist.shape == (5,) and dist.dtype == np.float64
    assert dist[0] == truth["k1"] and dist[1] == truth["k2"]
    assert dist[2] == 0.0 and dist[3] == 0.0 and dist[4] == 0.0
    rvec = cv["rvec"]; tvec = cv["tvec"]
    assert rvec.shape == (3,) and tvec.shape == (3,)
    assert np.allclose(tvec, 0.0)
    R = cv["rotation_matrix"]
    assert R.shape == (3, 3)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    print("[OK] to_opencv_intrinsics produces well-formed K, dist, rvec, tvec")


def test_opencv_projection_matches_native_model():
    """A celestial direction must project to the same pixel via the native
    model and via the OpenCV-style (K, dist, rvec, tvec) pipeline.
    """
    res, truth, native_pixels, az_el = _make_synth_result()
    cv = ox.to_opencv_intrinsics(res)
    K, dist, rvec, tvec = cv["camera_matrix"], cv["dist_coeffs"], cv["rvec"], cv["tvec"]

    max_err = 0.0
    for (az, el), (u_native, v_native) in zip(az_el, native_pixels):
        d = cc.enu_unit_vector(az, el)
        u_cv, v_cv = ox.project_with_opencv_model(d, K, dist, rvec, tvec)
        e = math.hypot(u_native - u_cv, v_native - v_cv)
        max_err = max(max_err, e)
    assert max_err < 1e-9, f"native vs opencv-model max pixel error = {max_err}"
    print(f"[OK] OpenCV-style projection matches native model (max err = {max_err:.2e} px)")


def test_export_yaml_contains_expected_values():
    res, truth, _, _ = _make_synth_result()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.yaml")
        ox.export_opencv_yaml(res, p)
        with open(p) as f: txt = f.read()
        assert txt.startswith("%YAML:1.0")
        assert "image_width: 1280" in txt
        assert "image_height: 720" in txt
        assert "camera_matrix:" in txt
        assert "distortion_coefficients:" in txt
        assert "!!opencv-matrix" in txt
        # focal length should appear in camera_matrix data
        assert "950" in txt
        # k1 negative
        assert "-0.07" in txt
    print("[OK] OpenCV YAML export contains expected fields and values")


def test_export_json_contains_expected_values():
    res, truth, _, _ = _make_synth_result()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.json")
        ox.export_opencv_json(res, p)
        with open(p) as f: data = json.load(f)
        assert data["image_width"] == 1280 and data["image_height"] == 720
        K = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
        assert K[0, 0] == truth["f"] and K[1, 1] == truth["f"]
        dist = data["distortion_coefficients"]["data"]
        assert dist[0] == truth["k1"]
        assert dist[1] == truth["k2"]
        assert dist[2:] == [0.0, 0.0, 0.0]
        # Rotation vector round-trips back to the rotation matrix
        rvec = np.array(data["rotation_vector"]["data"])
        R = ox.matrix_from_rodrigues(rvec)
        R_native = cc.rotation_world_to_cam(
            math.radians(truth["yaw_deg"]),
            math.radians(truth["pitch_deg"]),
            math.radians(truth["roll_deg"]),
        )
        assert np.allclose(R, R_native, atol=1e-9)
    print("[OK] OpenCV JSON export round-trips through Rodrigues correctly")


def test_export_npz_loads_back():
    res, truth, _, _ = _make_synth_result()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.npz")
        ox.export_numpy_npz(res, p)
        loaded = np.load(p)
        K = loaded["camera_matrix"]
        dist = loaded["dist_coeffs"]
        size = loaded["image_size"]
        assert K.shape == (3, 3)
        assert dist.shape == (5,)
        assert tuple(size) == (1280, 720)
        assert K[0, 0] == truth["f"]
        assert dist[0] == truth["k1"]
    print("[OK] NPZ export loads back via numpy.load")


def test_export_ros_camera_info():
    res, truth, _, _ = _make_synth_result()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "camera_info.yaml")
        ox.export_ros_camera_info(res, p)
        with open(p) as f: txt = f.read()
        assert "distortion_model: plumb_bob" in txt
        assert "image_width: 1280" in txt
        assert "image_height: 720" in txt
        assert "rectification_matrix:" in txt
        assert "projection_matrix:" in txt
    print("[OK] ROS camera_info YAML contains expected fields")


def test_optional_cv2_roundtrip():
    """If cv2 is installed, verify the exported YAML loads and produces the
    same projections via cv2.projectPoints."""
    try:
        import cv2  # type: ignore
    except ImportError:
        print("[SKIP] cv2 not installed; skipping cv2 round-trip test")
        return
    res, truth, native_pixels, az_el = _make_synth_result()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.yaml")
        ox.export_opencv_yaml(res, p)
        fs = cv2.FileStorage(p, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
        rvec = fs.getNode("rotation_vector").mat()
        tvec = fs.getNode("translation_vector").mat()
        fs.release()
        assert K.shape == (3, 3)
        assert dist.size == 5
        # Project via cv2 and compare to native pixels
        directions = np.array([cc.enu_unit_vector(az, el) for az, el in az_el],
                              dtype=np.float64).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(directions, rvec, tvec, K, dist.reshape(-1))
        proj = proj.reshape(-1, 2)
        err = max(math.hypot(proj[i, 0] - native_pixels[i][0],
                             proj[i, 1] - native_pixels[i][1])
                  for i in range(len(native_pixels)))
        assert err < 1e-6, f"cv2.projectPoints disagrees with native by {err}"
    print(f"[OK] cv2 round-trip via FileStorage + projectPoints (max err = {err:.2e} px)")


# ---------------------------------------------------------------------------
def main():
    print(">>> opencv_export tests")
    test_rodrigues_roundtrip()
    test_to_opencv_intrinsics_shapes_and_values()
    test_opencv_projection_matches_native_model()
    test_export_yaml_contains_expected_values()
    test_export_json_contains_expected_values()
    test_export_npz_loads_back()
    test_export_ros_camera_info()
    test_optional_cv2_roundtrip()
    print("\nAll opencv_export tests passed.")


if __name__ == "__main__":
    main()
