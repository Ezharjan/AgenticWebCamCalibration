"""Self-tests for the celestial calibration solver.

Run inside the project conda environment:

    conda activate cg
    python test_celestial_calibration.py

The forward-model section uses only numpy and runs everywhere.
The round-trip section uses scipy when available; otherwise it falls back
to a small pure-numpy Levenberg-Marquardt implementation built into
celestial_calibration.py.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone

import numpy as np

import celestial_calibration as cc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _approx(a, b, tol=1e-6, name=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: {a} != {b} (tol {tol})")


def _approx_pair(p, q, tol=1e-6, name=""):
    _approx(p[0], q[0], tol, f"{name}.x")
    _approx(p[1], q[1], tol, f"{name}.y")


# ---------------------------------------------------------------------------
# Forward-model unit tests
# ---------------------------------------------------------------------------
def test_enu_unit_vector():
    v = cc.enu_unit_vector(0.0, 0.0)
    _approx_pair((v[0], v[1]), (0.0, 1.0), 1e-9, "north_xy")
    _approx(v[2], 0.0, 1e-9, "north_z")
    v = cc.enu_unit_vector(90.0, 0.0)
    _approx(v[0], 1.0, 1e-9, "east_x"); _approx(v[1], 0.0, 1e-9, "east_y"); _approx(v[2], 0.0, 1e-9, "east_z")
    v = cc.enu_unit_vector(123.0, 90.0)
    _approx(v[0], 0.0, 1e-9, "zen_x"); _approx(v[1], 0.0, 1e-9, "zen_y"); _approx(v[2], 1.0, 1e-9, "zen_z")
    print("[OK] enu_unit_vector basic directions")


def test_zero_rotation_north_to_optical_axis():
    f = 1000.0
    cx, cy = 320.0, 240.0
    p = cc.project_az_el(0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, cx, cy)
    assert p is not None
    _approx_pair(p, (cx, cy), 1e-6, "north_at_zero")
    print("[OK] zero rotation: North at horizon -> principal point")


def test_yaw_swings_east_to_center():
    f = 800.0
    cx, cy = 400.0, 300.0
    yaw = math.radians(90.0)
    p = cc.project_az_el(90.0, 0.0, f, yaw, 0.0, 0.0, 0.0, 0.0, cx, cy)
    assert p is not None
    _approx_pair(p, (cx, cy), 1e-6, "yaw_east")
    print("[OK] yaw=+90 deg places az=90 body at principal point")


def test_pitch_lifts_zenith_to_center():
    f = 500.0
    cx, cy = 100.0, 100.0
    pitch = math.radians(90.0)
    p = cc.project_az_el(42.0, 90.0, f, 0.0, pitch, 0.0, 0.0, 0.0, cx, cy)
    assert p is not None
    _approx_pair(p, (cx, cy), 1e-6, "pitch_zenith")
    print("[OK] pitch=+90 deg places zenith at principal point")


def test_focal_scales_offset_correctly():
    f = 1000.0
    cx, cy = 0.0, 0.0
    p = cc.project_az_el(1.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, cx, cy)
    expected_x = f * math.tan(math.radians(1.0))
    _approx(p[0], expected_x, 1e-4, "focal_offset_x")
    _approx(p[1], 0.0, 1e-6, "focal_offset_y")
    print("[OK] focal length scales angular offset by tan(theta)")


def test_distortion_radial_only():
    f = 1000.0
    cx, cy = 0.0, 0.0
    k1, k2 = 0.1, 0.0
    az = 5.0
    p_no = cc.project_az_el(az, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, cx, cy)
    p_d  = cc.project_az_el(az, 0.0, f, 0.0, 0.0, 0.0, k1, k2, cx, cy)
    x_n = p_no[0] / f
    factor = 1 + k1 * (x_n ** 2)
    _approx(p_d[0], p_no[0] * factor, 1e-4, "distorted_x")
    _approx(p_d[1], 0.0, 1e-6, "distorted_y")
    print("[OK] radial distortion applied along radial direction")


def test_residuals_zero_at_truth():
    truth = dict(f=1200.0, yaw=math.radians(35.0), pitch=math.radians(8.0),
                 roll=math.radians(-2.5), k1=-0.12, k2=0.04,
                 cx=640.0, cy=360.0)
    rng = np.random.default_rng(42)
    az_el = []
    pixels = []
    for _ in range(12):
        az = math.degrees(truth["yaw"]) + rng.uniform(-25, 25)
        el = math.degrees(truth["pitch"]) + rng.uniform(-15, 15)
        u, v = cc.project_az_el(az, el,
                                truth["f"], truth["yaw"], truth["pitch"], truth["roll"],
                                truth["k1"], truth["k2"], truth["cx"], truth["cy"])
        az_el.append([az, el])
        pixels.append([u, v])
    az_el = np.array(az_el)
    pixels = np.array(pixels)
    weights = np.ones(len(pixels))
    p = np.array([truth["f"], truth["yaw"], truth["pitch"],
                  truth["roll"], truth["k1"], truth["k2"]])
    r = cc._residuals(p, az_el, pixels, weights, truth["cx"], truth["cy"])
    max_abs = float(np.max(np.abs(r)))
    assert max_abs < 1e-6, f"residuals not zero at truth: max={max_abs}"
    print(f"[OK] residuals at truth max |r| = {max_abs:.2e}")


# ---------------------------------------------------------------------------
# Round-trip optimiser tests
# ---------------------------------------------------------------------------
def _generate_synthetic_observations(truth, image_w, image_h, n=24, noise_px=0.0, seed=7):
    """Sample observations distributed across the FOV (corners + interior).

    Wide angular spread is essential to disentangle f from k1/k2 in calibration.
    """
    cx, cy = image_w / 2.0, image_h / 2.0
    rng = np.random.default_rng(seed)
    fov_h = 2 * math.degrees(math.atan((image_w / 2.0) / truth["f"]))
    fov_v = 2 * math.degrees(math.atan((image_h / 2.0) / truth["f"]))
    side = max(3, int(math.ceil(math.sqrt(n))))
    pts = []
    for ix in range(side):
        for iy in range(side):
            fx = (ix + 0.5) / side - 0.5
            fy = (iy + 0.5) / side - 0.5
            d_az = fx * 0.95 * fov_h
            d_el = fy * 0.95 * fov_v
            pts.append((d_az, d_el))
    rng.shuffle(pts)
    pts = pts[:n]
    base_az = truth["yaw_deg"]
    base_el = truth["pitch_deg"]
    obs_list = []
    for i, (d_az, d_el) in enumerate(pts):
        az = base_az + d_az
        el = base_el + d_el
        u, v = cc.project_az_el(
            az, el, truth["f"],
            math.radians(truth["yaw_deg"]),
            math.radians(truth["pitch_deg"]),
            math.radians(truth["roll_deg"]),
            truth["k1"], truth["k2"], cx, cy,
        )
        if noise_px > 0:
            u += rng.normal(0.0, noise_px)
            v += rng.normal(0.0, noise_px)
        obs_list.append(cc.Observation(
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            pixel_x=float(u), pixel_y=float(v),
            body=f"synthetic_{i}",
            az_deg=float(az), el_deg=float(el),
        ))
    return obs_list


def test_roundtrip_solver_noise_free():
    if not cc.HAS_SCIPY:
        print("  (scipy unavailable - using pure-numpy LM fallback)")
    truth = dict(
        f=900.0, yaw_deg=215.0, pitch_deg=12.0, roll_deg=-3.5,
        k1=-0.08, k2=0.02,
    )
    W, H = 1280, 720
    obs_list = _generate_synthetic_observations(truth, W, H, n=25, noise_px=0.0, seed=11)
    res = cc.solve_calibration(obs_list, image_width=W, image_height=H, max_nfev=500)
    print(f"  [noise-free, N={res.n_observations}]")
    print(f"  truth   f={truth['f']:.4f}  yaw={truth['yaw_deg']:.4f}  "
          f"pitch={truth['pitch_deg']:.4f}  roll={truth['roll_deg']:.4f}  "
          f"k1={truth['k1']:.4f}  k2={truth['k2']:.4f}")
    print(f"  solved  f={res.focal_length_px:.4f}  yaw={res.yaw_deg:.4f}  "
          f"pitch={res.pitch_deg:.4f}  roll={res.roll_deg:.4f}  "
          f"k1={res.k1:.4f}  k2={res.k2:.4f}")
    print(f"  RMS reprojection error: {res.rms_reprojection_error_px:.2e} px")
    assert abs(res.focal_length_px - truth["f"])  < 0.05, f"f off: {res.focal_length_px}"
    assert abs(res.yaw_deg   - truth["yaw_deg"])   < 1e-4
    assert abs(res.pitch_deg - truth["pitch_deg"]) < 1e-4
    assert abs(res.roll_deg  - truth["roll_deg"])  < 1e-4
    assert abs(res.k1 - truth["k1"]) < 1e-4
    assert abs(res.k2 - truth["k2"]) < 1e-3
    assert res.rms_reprojection_error_px < 1e-3
    print("[OK] noise-free round-trip recovers ground truth to ~machine precision")


def test_roundtrip_solver_with_noise():
    if not cc.HAS_SCIPY:
        print("  (scipy unavailable - using pure-numpy LM fallback)")
    truth = dict(
        f=900.0, yaw_deg=215.0, pitch_deg=12.0, roll_deg=-3.5,
        k1=-0.08, k2=0.02,
    )
    W, H = 1280, 720
    obs_list = _generate_synthetic_observations(truth, W, H, n=25, noise_px=0.3, seed=11)
    res = cc.solve_calibration(obs_list, image_width=W, image_height=H, max_nfev=500)
    print(f"  [sigma=0.3 px, N={res.n_observations}]")
    print(f"  solved  f={res.focal_length_px:.3f}  yaw={res.yaw_deg:.4f}  "
          f"pitch={res.pitch_deg:.4f}  roll={res.roll_deg:.4f}  "
          f"k1={res.k1:.4f}  k2={res.k2:.4f}")
    print(f"  RMS reprojection error: {res.rms_reprojection_error_px:.3f} px")
    assert abs(res.focal_length_px - truth["f"])  < 3.0
    assert abs(res.yaw_deg   - truth["yaw_deg"])   < 0.05
    assert abs(res.pitch_deg - truth["pitch_deg"]) < 0.05
    assert abs(res.roll_deg  - truth["roll_deg"])  < 0.10
    assert abs(res.k1 - truth["k1"]) < 0.02
    assert abs(res.k2 - truth["k2"]) < 0.05
    assert res.rms_reprojection_error_px < 1.0
    print("[OK] noisy round-trip recovers ground truth within tolerance")


def test_too_few_observations_rejected():
    obs = [cc.Observation(
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        pixel_x=100.0, pixel_y=100.0, body="x",
        az_deg=0.0, el_deg=0.0,
    )] * 5
    try:
        cc.solve_calibration(obs, 640, 480)
    except ValueError as e:
        assert "Need >=6" in str(e), e
        print("[OK] solver rejects <6 observations")
        return
    raise AssertionError("solver should have rejected <6 observations")


def test_csv_loader_roundtrip():
    import tempfile
    p = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    p.write("timestamp_utc,pixel_x,pixel_y,body,az_deg,el_deg\n")
    p.write("2026-05-01T12:00:00Z,640.5,360.0,sun,180.0,55.0\n")
    p.write("2026-05-01T12:05:00Z,700.5,360.0,moon,182.0,55.0\n")
    p.close()
    obs = cc.load_observations_csv(p.name)
    assert len(obs) == 2
    assert obs[0].body == "sun"
    assert obs[0].az_deg == 180.0
    assert obs[0].timestamp.tzinfo is not None
    print("[OK] csv loader parses ISO-8601 + optional az/el")


def main():
    print(">>> Forward-model unit tests")
    test_enu_unit_vector()
    test_zero_rotation_north_to_optical_axis()
    test_yaw_swings_east_to_center()
    test_pitch_lifts_zenith_to_center()
    test_focal_scales_offset_correctly()
    test_distortion_radial_only()
    test_residuals_zero_at_truth()
    test_too_few_observations_rejected()
    test_csv_loader_roundtrip()
    print(">>> Round-trip optimiser tests")
    test_roundtrip_solver_noise_free()
    test_roundtrip_solver_with_noise()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
