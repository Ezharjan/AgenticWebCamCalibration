"""Celestial Camera Calibration Solver
======================================

Given >=6 observations of celestial bodies (sun, moon, planets, named stars)
at known UTC timestamps with their pixel positions in a webcam image, solve
for the camera intrinsics and orientation:

  * focal_length_px  - pinhole focal length (px), square pixels
  * yaw, pitch, roll - extrinsic rotation (radians/degrees)
  * k1, k2           - Brown-Conrady radial distortion coefficients

Camera model
------------
  * Pinhole + radial distortion (k1, k2).
  * Square pixels, single focal length f.
  * Principal point (cx, cy) fixed at image center (configurable).
  * Right-handed camera frame: +x right, +y down, +z forward (optical axis).

World model
-----------
  * Local ENU (East-North-Up) frame at camera location.
  * Celestial direction is a unit vector built from (azimuth, elevation).
  * Reference orientation (yaw=pitch=roll=0): the optical axis points North,
    camera +x points East, camera +y points Down.
  * yaw   - rotation about Up (compass: positive = swing from North toward East)
  * pitch - tilt about camera right axis      (positive = look up above horizon)
  * roll  - rotation about optical axis       (positive = image content CCW)

The forward projection of an ENU unit vector v is:
    v_cam      = R_roll(roll) @ R_yaw_pitch(yaw, pitch) @ v
    x_n, y_n   = v_cam[0]/v_cam[2], v_cam[1]/v_cam[2]
    r2         = x_n^2 + y_n^2
    x_d, y_d   = (1 + k1 r2 + k2 r2^2) * (x_n, y_n)
    u, v_pix   = (f*x_d + cx, f*y_d + cy)

The solver minimises sum_i (u_i - u_obs_i)^2 + (v_i - v_obs_i)^2 in pixel space
using scipy.optimize.least_squares (Levenberg-Marquardt / Trust-Region).

Run inside the project's conda environment:
    conda activate cg
    python celestial_calibration.py --help
"""
from __future__ import annotations

import os
import sys
import math
import csv
import json
import argparse
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Sequence, List, Tuple, Dict, Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import least_squares  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Tiny pure-numpy Levenberg-Marquardt fallback (used only if scipy is missing).
# scipy.optimize.least_squares is preferred because it has battle-tested
# bounds handling, scaling, and stopping criteria. This fallback exists so
# that the unit tests can exercise the round-trip even in environments
# without scipy.
# ---------------------------------------------------------------------------
def _lm_least_squares_numpy(fun, x0, lb, ub, max_iter=200, tol=1e-10):
    """Bound-constrained Levenberg-Marquardt for small problems.

    fun(x) -> 1D residual vector. Jacobian is computed via central differences.
    Returns an object with .x, .cost, .success.
    """
    class _R: pass
    x = np.clip(np.asarray(x0, dtype=float).copy(), lb, ub)
    r = fun(x)
    cost = 0.5 * float(np.dot(r, r))
    lam = 1e-3
    eps = 1e-7
    n = len(x)
    success = False
    for _ in range(max_iter):
        # Central-difference Jacobian
        J = np.zeros((len(r), n))
        for j in range(n):
            step = max(eps, eps * (abs(x[j]) + 1.0))
            xp = x.copy(); xm = x.copy()
            xp[j] = min(ub[j], x[j] + step)
            xm[j] = max(lb[j], x[j] - step)
            actual_step = xp[j] - xm[j]
            if actual_step <= 0:
                continue
            J[:, j] = (fun(xp) - fun(xm)) / actual_step
        JTJ = J.T @ J
        JTr = J.T @ r
        # LM step: (JTJ + lam * diag(JTJ)) dx = -JTr
        diag = np.diag(np.diag(JTJ))
        try:
            dx = np.linalg.solve(JTJ + lam * diag + 1e-12 * np.eye(n), -JTr)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        x_new = np.clip(x + dx, lb, ub)
        r_new = fun(x_new)
        cost_new = 0.5 * float(np.dot(r_new, r_new))
        if cost_new < cost:
            if abs(cost - cost_new) < tol * max(1.0, cost):
                x = x_new; r = r_new; cost = cost_new
                success = True
                break
            x = x_new; r = r_new; cost = cost_new
            lam = max(lam / 5, 1e-9)
        else:
            lam *= 5
            if lam > 1e12:
                break
    out = _R()
    out.x = x
    out.cost = cost
    out.success = success
    return out

try:
    from skyfield.api import load, wgs84, Star  # type: ignore
    from skyfield.units import Angle  # type: ignore
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("celestial_calibration")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)


# ---------------------------------------------------------------------------
# Named-star catalog (J2000 RA/Dec, ICRS)
# ---------------------------------------------------------------------------
# RA in hours, Dec in degrees. Source: SIMBAD/Hipparcos (rounded).
NAMED_STARS: Dict[str, Tuple[float, float]] = {
    "polaris":     (2.5301944, 89.2641111),
    "sirius":      (6.7524806, -16.7161083),
    "canopus":     (6.3992194, -52.6957639),
    "arcturus":   (14.2610278,  19.1824194),
    "vega":       (18.6156472,  38.7836889),
    "capella":     (5.2781944,  45.9979889),
    "rigel":       (5.2422778,  -8.2016639),
    "procyon":     (7.6550278,   5.2249944),
    "betelgeuse":  (5.9195297,   7.4070611),
    "altair":     (19.8463889,   8.8683417),
    "aldebaran":   (4.5986750,  16.5092583),
    "antares":    (16.4901222, -26.4319944),
    "spica":      (13.4198806, -11.1613083),
    "pollux":      (7.7552611,  28.0261083),
    "deneb":      (20.6905333,  45.2803639),
    "regulus":    (10.1395306,  11.9671722),
    "fomalhaut":  (22.9608389, -29.6222778),
    "mira":        (2.3228056,  -2.9776694),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single (timestamp, pixel, celestial body) observation.

    Provide either a `body` name (resolved via skyfield) OR an explicit
    (az_deg, el_deg). Explicit angles take precedence and bypass skyfield.
    """
    timestamp: datetime          # tz-aware UTC
    pixel_x: float
    pixel_y: float
    body: str                    # e.g. "sun", "moon", "venus", "polaris"
    az_deg: Optional[float] = None
    el_deg: Optional[float] = None
    weight: float = 1.0          # per-observation weight (default 1)

    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class CalibrationResult:
    focal_length_px: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    k1: float
    k2: float
    image_width: int
    image_height: int
    cx: float
    cy: float
    rms_reprojection_error_px: float
    max_reprojection_error_px: float
    median_reprojection_error_px: float
    n_observations: int
    converged: bool
    cost: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    per_obs_residuals_px: List[Tuple[float, float]]
    per_obs_predicted_px:  List[Tuple[float, float]]
    per_obs_observed_px:   List[Tuple[float, float]]
    per_obs_az_el_deg:     List[Tuple[float, float]]
    per_obs_body:          List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Round numeric fields for cleaner JSON output
        for k in (
            "focal_length_px", "yaw_deg", "pitch_deg", "roll_deg",
            "k1", "k2", "rms_reprojection_error_px",
            "max_reprojection_error_px", "median_reprojection_error_px",
            "fov_horizontal_deg", "fov_vertical_deg", "cost",
        ):
            d[k] = float(round(d[k], 6))
        return d


# ---------------------------------------------------------------------------
# Forward projection model (pure numpy; differentiable via finite diff)
# ---------------------------------------------------------------------------

def enu_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit direction vector in local ENU from compass azimuth/elevation."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array([
        math.sin(az) * math.cos(el),  # East
        math.cos(az) * math.cos(el),  # North
        math.sin(el),                 # Up
    ], dtype=np.float64)


def rotation_world_to_cam(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """3x3 rotation taking ENU vectors to camera frame (x right, y down, z fwd).

    yaw, pitch, roll are in radians.
    """
    sy, cy = math.sin(yaw),   math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll),  math.cos(roll)

    # Camera basis in ENU at zero-roll, derived from azimuth/elevation of
    # the optical axis. Rows of R_no_roll are (x_cam, y_cam, z_cam) in ENU.
    R_no_roll = np.array([
        [cy,        -sy,        0.0],   # x_cam = right
        [sp*sy,      sp*cy,    -cp],    # y_cam = down
        [sy*cp,      cy*cp,     sp],    # z_cam = forward (optical axis)
    ], dtype=np.float64)

    # Roll about camera +z (applied in camera frame after yaw/pitch).
    R_roll = np.array([
        [ cr,  sr, 0.0],
        [-sr,  cr, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return R_roll @ R_no_roll


def project_az_el(
    az_deg: float, el_deg: float,
    f: float, yaw: float, pitch: float, roll: float,
    k1: float, k2: float,
    cx: float, cy: float,
) -> Optional[Tuple[float, float]]:
    """Project a celestial body at (az, el) to pixel coordinates.

    Returns (u, v) or None if behind the camera (z_cam <= 0).
    Angles for yaw/pitch/roll are in radians.
    """
    v_world = enu_unit_vector(az_deg, el_deg)
    R = rotation_world_to_cam(yaw, pitch, roll)
    v_cam = R @ v_world
    z = v_cam[2]
    if z <= 1e-9:
        return None
    x_n = v_cam[0] / z
    y_n = v_cam[1] / z
    r2 = x_n * x_n + y_n * y_n
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    return f * x_n * factor + cx, f * y_n * factor + cy


# ---------------------------------------------------------------------------
# Skyfield body lookup
# ---------------------------------------------------------------------------

class BodyResolver:
    """Resolves (timestamp, body name) -> (azimuth_deg, elevation_deg).

    Uses skyfield with the JPL DE421 ephemeris for solar-system bodies,
    and a small named-star catalog for fixed stars.
    """

    def __init__(
        self,
        latitude_deg: float,
        longitude_deg: float,
        elevation_m: float = 0.0,
        ephemeris: str = "de421.bsp",
    ):
        if not HAS_SKYFIELD:
            raise RuntimeError(
                "skyfield is not installed. Install with `pip install skyfield` "
                "or use az_deg/el_deg directly on each Observation."
            )
        self.lat = latitude_deg
        self.lon = longitude_deg
        self.elevation_m = elevation_m
        self._ts = load.timescale()
        self._eph = load(ephemeris)
        self._earth = self._eph["earth"]
        self._site = self._earth + wgs84.latlon(latitude_deg, longitude_deg, elevation_m=elevation_m)

    def _resolve_target(self, body_name: str):
        name = body_name.strip().lower()
        # Solar-system bodies
        ssb_map = {
            "sun":     "sun",
            "moon":    "moon",
            "mercury": "mercury",
            "venus":   "venus",
            "mars":    "mars barycenter",
            "jupiter": "jupiter barycenter",
            "saturn":  "saturn barycenter",
            "uranus":  "uranus barycenter",
            "neptune": "neptune barycenter",
        }
        if name in ssb_map:
            return self._eph[ssb_map[name]]
        if name in NAMED_STARS:
            ra_h, dec_d = NAMED_STARS[name]
            return Star(ra_hours=ra_h, dec_degrees=dec_d)
        raise KeyError(f"Unknown celestial body: {body_name!r}. "
                       f"Supported: {sorted(list(ssb_map) + list(NAMED_STARS))}")

    def az_el(self, when: datetime, body_name: str) -> Tuple[float, float]:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        t = self._ts.from_datetime(when)
        target = self._resolve_target(body_name)
        astrometric = self._site.at(t).observe(target).apparent()
        alt, az, _ = astrometric.altaz()
        return float(az.degrees), float(alt.degrees)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

# Parameter ordering: [f, yaw, pitch, roll, k1, k2]
PARAM_NAMES = ("focal_length_px", "yaw_rad", "pitch_rad", "roll_rad", "k1", "k2")


def _residuals(
    params: np.ndarray,
    az_el_array: np.ndarray,        # (N, 2) az_deg, el_deg
    pixels: np.ndarray,             # (N, 2) u, v
    weights: np.ndarray,            # (N,)
    cx: float, cy: float,
    behind_penalty: float = 1.0e4,
) -> np.ndarray:
    f, yaw, pitch, roll, k1, k2 = params
    R = rotation_world_to_cam(yaw, pitch, roll)
    res = np.zeros(2 * len(az_el_array), dtype=np.float64)
    for i, (az, el) in enumerate(az_el_array):
        v = enu_unit_vector(az, el)
        v_cam = R @ v
        z = v_cam[2]
        if z <= 1e-6:
            # Body behind camera -> large but finite residual
            res[2 * i]     = behind_penalty * weights[i]
            res[2 * i + 1] = behind_penalty * weights[i]
            continue
        x_n = v_cam[0] / z
        y_n = v_cam[1] / z
        r2  = x_n * x_n + y_n * y_n
        fac = 1.0 + k1 * r2 + k2 * r2 * r2
        u_pred = f * x_n * fac + cx
        v_pred = f * y_n * fac + cy
        res[2 * i]     = weights[i] * (u_pred - pixels[i, 0])
        res[2 * i + 1] = weights[i] * (v_pred - pixels[i, 1])
    return res


def _initial_guess(
    az_el_array: np.ndarray,
    pixels: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Pick a sane initial guess using the observation nearest the image center.

    yaw <- az of central body, pitch <- el of central body, roll <- 0,
    f <- max(W, H) (~ 50-deg horizontal FOV), k1 = k2 = 0.
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    d2 = (pixels[:, 0] - cx) ** 2 + (pixels[:, 1] - cy) ** 2
    i = int(np.argmin(d2))
    yaw0   = math.radians(az_el_array[i, 0])
    pitch0 = math.radians(az_el_array[i, 1])
    f0     = float(max(image_width, image_height))
    return np.array([f0, yaw0, pitch0, 0.0, 0.0, 0.0], dtype=np.float64)


def solve_calibration(
    observations: Sequence[Observation],
    image_width: int,
    image_height: int,
    *,
    camera_lat_deg: Optional[float] = None,
    camera_lon_deg: Optional[float] = None,
    camera_elevation_m: float = 0.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    initial_guess: Optional[Sequence[float]] = None,
    fix_distortion: bool = False,
    fix_roll: bool = False,
    max_nfev: int = 200,
    verbose: int = 0,
) -> CalibrationResult:
    """Solve for camera intrinsics + orientation from celestial observations.

    Parameters
    ----------
    observations : sequence of Observation (>= 6 required)
    image_width, image_height : pixel dimensions
    camera_lat_deg, camera_lon_deg : camera location (required if any
        observation lacks explicit az/el and skyfield must be used)
    camera_elevation_m : optional altitude for skyfield (default 0)
    cx, cy : principal point; defaults to image center
    initial_guess : optional [f, yaw_rad, pitch_rad, roll_rad, k1, k2]
    fix_distortion : if True, k1 = k2 = 0 (4-parameter solve)
    fix_roll : if True, roll = 0 (5-parameter solve, or 3 with fix_distortion)
    max_nfev : max function evaluations for the optimiser
    verbose : pass-through to scipy least_squares (0/1/2)
    """
    n = len(observations)
    if n < 6:
        raise ValueError(f"Need >=6 observations to solve, got {n}.")

    if not HAS_SCIPY:
        logger.warning(
            "scipy not available - falling back to a small pure-numpy LM "
            "solver. Install scipy for production use (`pip install scipy` "
            "or run inside the project's conda env: `conda activate cg`)."
        )

    if cx is None:
        cx = image_width / 2.0
    if cy is None:
        cy = image_height / 2.0

    # Resolve az/el for each observation (skyfield or pre-computed)
    az_el = np.zeros((n, 2), dtype=np.float64)
    pixels = np.zeros((n, 2), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    bodies: List[str] = []
    resolver: Optional[BodyResolver] = None

    needs_resolver = any(o.az_deg is None or o.el_deg is None for o in observations)
    if needs_resolver:
        if camera_lat_deg is None or camera_lon_deg is None:
            raise ValueError(
                "Camera latitude/longitude required to resolve celestial bodies "
                "(or pre-fill az_deg/el_deg on every Observation)."
            )
        resolver = BodyResolver(camera_lat_deg, camera_lon_deg, camera_elevation_m)

    for i, o in enumerate(observations):
        if o.az_deg is not None and o.el_deg is not None:
            az_d, el_d = float(o.az_deg), float(o.el_deg)
        else:
            assert resolver is not None
            az_d, el_d = resolver.az_el(o.timestamp, o.body)
        az_el[i] = (az_d, el_d)
        pixels[i] = (o.pixel_x, o.pixel_y)
        weights[i] = max(1e-6, float(o.weight))
        bodies.append(o.body)

    # Initial guess
    if initial_guess is None:
        x0 = _initial_guess(az_el, pixels, image_width, image_height)
    else:
        x0 = np.array(initial_guess, dtype=np.float64)
        if x0.shape != (6,):
            raise ValueError("initial_guess must be a length-6 sequence.")

    # Bounds: keep f positive, angles within +-2pi, distortion bounded
    lb = np.array([1.0,    -2*math.pi, -math.pi/2, -math.pi, -2.0, -2.0])
    ub = np.array([1e6,     2*math.pi,  math.pi/2,  math.pi,  2.0,  2.0])

    # If we're fixing parameters, lock them by clamping bounds at the initial value
    if fix_distortion:
        x0[4] = 0.0
        x0[5] = 0.0
        lb[4] = -1e-12; ub[4] = 1e-12
        lb[5] = -1e-12; ub[5] = 1e-12
    if fix_roll:
        x0[3] = 0.0
        lb[3] = -1e-12; ub[3] = 1e-12

    # Make sure x0 is inside bounds
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)

    fun = lambda p: _residuals(p, az_el, pixels, weights, cx, cy)

    if HAS_SCIPY:
        result = least_squares(
            fun, x0, bounds=(lb, ub),
            method="trf", x_scale="jac",
            max_nfev=max_nfev, verbose=verbose,
        )
    else:
        result = _lm_least_squares_numpy(fun, x0, lb, ub, max_iter=max_nfev)

    f_, yaw_, pitch_, roll_, k1_, k2_ = result.x

    # Recompute residuals with weights=1 to report unweighted pixel errors
    raw_res = _residuals(result.x, az_el, pixels, np.ones(n), cx, cy)
    raw_res = raw_res.reshape(n, 2)
    per_pt_err = np.linalg.norm(raw_res, axis=1)
    rms = float(np.sqrt(np.mean(per_pt_err ** 2)))
    mx  = float(np.max(per_pt_err))
    med = float(np.median(per_pt_err))

    # Predicted pixels for output
    predicted: List[Tuple[float, float]] = []
    for i in range(n):
        p = project_az_el(
            float(az_el[i, 0]), float(az_el[i, 1]),
            f_, yaw_, pitch_, roll_, k1_, k2_, cx, cy,
        )
        predicted.append((float("nan"), float("nan")) if p is None else (float(p[0]), float(p[1])))

    fov_h = 2 * math.degrees(math.atan((image_width  / 2.0) / f_))
    fov_v = 2 * math.degrees(math.atan((image_height / 2.0) / f_))

    return CalibrationResult(
        focal_length_px=float(f_),
        yaw_deg=math.degrees(float(yaw_)),
        pitch_deg=math.degrees(float(pitch_)),
        roll_deg=math.degrees(float(roll_)),
        k1=float(k1_),
        k2=float(k2_),
        image_width=int(image_width),
        image_height=int(image_height),
        cx=float(cx),
        cy=float(cy),
        rms_reprojection_error_px=rms,
        max_reprojection_error_px=mx,
        median_reprojection_error_px=med,
        n_observations=n,
        converged=bool(result.success),
        cost=float(result.cost),
        fov_horizontal_deg=fov_h,
        fov_vertical_deg=fov_v,
        per_obs_residuals_px=[(float(r[0]), float(r[1])) for r in raw_res],
        per_obs_predicted_px=predicted,
        per_obs_observed_px=[(float(p[0]), float(p[1])) for p in pixels],
        per_obs_az_el_deg=[(float(a), float(e)) for a, e in az_el],
        per_obs_body=bodies,
    )


# ---------------------------------------------------------------------------
# CSV input loader
# ---------------------------------------------------------------------------

REQUIRED_COLS = ("timestamp_utc", "pixel_x", "pixel_y", "body")
OPTIONAL_COLS = ("az_deg", "el_deg", "weight")


def load_observations_csv(path: str) -> List[Observation]:
    """Load observations from a CSV file.

    Required columns: timestamp_utc, pixel_x, pixel_y, body
    Optional columns: az_deg, el_deg, weight
    timestamp_utc is parsed as ISO 8601; trailing 'Z' is treated as UTC.
    """
    obs: List[Observation] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {missing}. "
                             f"Required = {REQUIRED_COLS}; optional = {OPTIONAL_COLS}.")
        for i, row in enumerate(reader, start=2):
            ts_raw = row["timestamp_utc"].strip()
            if ts_raw.endswith("Z"):
                ts_raw = ts_raw[:-1] + "+00:00"
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError as e:
                raise ValueError(f"Row {i}: bad timestamp {row['timestamp_utc']!r}: {e}")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            def _opt_float(key):
                v = row.get(key, "")
                v = v.strip() if isinstance(v, str) else v
                return float(v) if v not in (None, "", "nan", "NaN") else None

            obs.append(Observation(
                timestamp=ts,
                pixel_x=float(row["pixel_x"]),
                pixel_y=float(row["pixel_y"]),
                body=row["body"].strip(),
                az_deg=_opt_float("az_deg"),
                el_deg=_opt_float("el_deg"),
                weight=float(row["weight"]) if (row.get("weight", "") or "").strip() else 1.0,
            ))
    return obs


# ---------------------------------------------------------------------------
# Validation overlay (PIL; no matplotlib dependency)
# ---------------------------------------------------------------------------

def render_validation_overlay(
    image_path: str,
    result: CalibrationResult,
    output_path: str,
    *,
    radius: int = 8,
    line_width: int = 2,
) -> str:
    """Draw observed pixels (green circles), predicted pixels (red crosses),
    a residual line connecting them, and a body label, on top of the image.

    Returns the output path.
    """
    from PIL import Image, ImageDraw, ImageFont  # local import to keep core light

    img = Image.open(image_path).convert("RGB")
    if (img.width, img.height) != (result.image_width, result.image_height):
        logger.warning(
            "Overlay image size %dx%d != calibration size %dx%d; "
            "the overlay will still be drawn but pixel coords assume the "
            "calibration size.",
            img.width, img.height, result.image_width, result.image_height,
        )

    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, img.width // 80))
    except Exception:
        font = ImageFont.load_default()

    GREEN = (0, 220, 80, 255)
    RED   = (240, 60, 60, 255)
    YELLOW = (255, 220, 60, 255)
    BLACK = (0, 0, 0, 200)

    for (uo, vo), (up, vp), body in zip(
        result.per_obs_observed_px, result.per_obs_predicted_px, result.per_obs_body
    ):
        # Observed = green circle
        draw.ellipse(
            [uo - radius, vo - radius, uo + radius, vo + radius],
            outline=GREEN, width=line_width,
        )
        # Predicted = red cross + small filled dot
        if not (math.isnan(up) or math.isnan(vp)):
            draw.line([up - radius, vp, up + radius, vp], fill=RED, width=line_width)
            draw.line([up, vp - radius, up, vp + radius], fill=RED, width=line_width)
            draw.ellipse([up - 2, vp - 2, up + 2, vp + 2], fill=RED)
            # Residual segment
            draw.line([uo, vo, up, vp], fill=YELLOW, width=1)

        label = body
        # Black outline + green fill for legibility
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((uo + radius + 4 + dx, vo - radius + dy),
                              label, fill=BLACK, font=font)
        draw.text((uo + radius + 4, vo - radius), label, fill=GREEN, font=font)

    # Header text
    header = (
        f"f={result.focal_length_px:.1f}px  "
        f"FOV_h={result.fov_horizontal_deg:.1f}deg  "
        f"yaw={result.yaw_deg:.2f}  pitch={result.pitch_deg:.2f}  "
        f"roll={result.roll_deg:.2f}  "
        f"k1={result.k1:.4f}  k2={result.k2:.4f}  "
        f"RMS={result.rms_reprojection_error_px:.2f}px  "
        f"max={result.max_reprojection_error_px:.2f}px  "
        f"N={result.n_observations}"
    )
    pad = 6
    bbox = draw.textbbox((pad, pad), header, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text((pad, pad), header, fill=(255, 255, 255, 255), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(
        description="Solve camera focal length, yaw/pitch/roll, k1, k2 from "
                    ">=6 (timestamp, pixel, celestial body) observations.",
    )
    p.add_argument("--observations", required=True,
                   help="CSV with columns timestamp_utc,pixel_x,pixel_y,body[,az_deg,el_deg,weight]")
    p.add_argument("--image-width", type=int, required=True)
    p.add_argument("--image-height", type=int, required=True)
    p.add_argument("--latitude", type=float, default=None,
                   help="Camera latitude in degrees (required unless every row has az_deg/el_deg).")
    p.add_argument("--longitude", type=float, default=None,
                   help="Camera longitude in degrees (required unless every row has az_deg/el_deg).")
    p.add_argument("--elevation", type=float, default=0.0,
                   help="Camera elevation in metres above WGS84 ellipsoid (default 0).")
    p.add_argument("--cx", type=float, default=None,
                   help="Principal point x (default: image_width / 2).")
    p.add_argument("--cy", type=float, default=None,
                   help="Principal point y (default: image_height / 2).")
    p.add_argument("--fix-distortion", action="store_true",
                   help="Solve with k1=k2=0 fixed.")
    p.add_argument("--fix-roll", action="store_true",
                   help="Solve with roll=0 fixed.")
    p.add_argument("--image", default=None,
                   help="Path to the calibration image. If supplied along with "
                        "--overlay, a validation overlay PNG is rendered.")
    p.add_argument("--overlay", default=None,
                   help="Output path for the validation overlay PNG.")
    p.add_argument("--json-out", default=None,
                   help="Write the full calibration result as JSON to this file.")
    p.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    args = p.parse_args()

    obs = load_observations_csv(args.observations)
    if len(obs) < 6:
        raise SystemExit(f"Need >=6 observations, got {len(obs)}")

    res = solve_calibration(
        obs,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_lat_deg=args.latitude,
        camera_lon_deg=args.longitude,
        camera_elevation_m=args.elevation,
        cx=args.cx, cy=args.cy,
        fix_distortion=args.fix_distortion,
        fix_roll=args.fix_roll,
        verbose=args.verbose,
    )

    print()
    print(" Celestial Calibration Result ".center(60, "="))
    print(f"  Observations      : {res.n_observations}")
    print(f"  Converged         : {res.converged}")
    print(f"  Cost (LSQ)        : {res.cost:.4f}")
    print(f"  RMS reproj error  : {res.rms_reprojection_error_px:.3f} px")
    print(f"  Median            : {res.median_reprojection_error_px:.3f} px")
    print(f"  Max               : {res.max_reprojection_error_px:.3f} px")
    print()
    print(f"  focal_length_px   : {res.focal_length_px:.3f}")
    print(f"  yaw_deg           : {res.yaw_deg:+.4f}")
    print(f"  pitch_deg         : {res.pitch_deg:+.4f}")
    print(f"  roll_deg          : {res.roll_deg:+.4f}")
    print(f"  k1                : {res.k1:+.6f}")
    print(f"  k2                : {res.k2:+.6f}")
    print(f"  FOV_h_deg         : {res.fov_horizontal_deg:.3f}")
    print(f"  FOV_v_deg         : {res.fov_vertical_deg:.3f}")
    print("=" * 60)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        print(f"  JSON result -> {args.json_out}")

    if args.image and args.overlay:
        out = render_validation_overlay(args.image, res, args.overlay)
        print(f"  Overlay     -> {out}")


if __name__ == "__main__":
    _cli()
