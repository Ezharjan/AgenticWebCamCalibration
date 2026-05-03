# Agentic Webcam Calibration Tool

## Overview
A three-agent pipeline that validates, repairs, and calibrates live webcam URLs
using Apple DepthPro — entirely free and local, with no paid APIs or keys.

In addition to the agentic pipeline, the repo ships a **closed-form
astrometric calibration solver** (`celestial_calibration.py`) that recovers
focal length, yaw/pitch/roll and radial distortion from observations of the
sun, moon, planets or named stars in a webcam image.

Each agent operates **agentically**: it reasons about failures, selects the best
strategy, retries with alternative approaches, and validates its own results.

The pipeline takes a raw CSV of webcams and produces a **single enriched CSV**
with health status, repaired URLs, and estimated camera intrinsics (focal length,
field of view).

## Architecture
```text
  webcam_list.csv
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  output/webcam_calibration_results.csv  (single output)      │
  │  progressively enriched by each agent                        │
  └──────────────────────────────────────────────────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐
  │  Agent 1   │    │    Agent 2      │    │    Agent 3       │
  │  Health    │──▶│    URL Repair    │──▶│    Calibration   │
  │  Check     │    │  (multi-strat)  │    │  (DepthPro +     │
  │            │    │                 │    │   self-correct)  │
  └────────────┘    └─────────────────┘    └──────────────────┘
   • Validate URL   • Re-validate        • Image quality check
   • Magic bytes    • Strip query params  • DepthPro inference
   • Retry on       • Scrape source page  • Plausibility check
     transient err  • Wayback Machine     • Retry: center crop
                    • Pattern mutations   • Retry: resize
                    • FAA-specific API    • Confidence scoring

  ────────────────────────────────────────────────────────────
  Optional: closed-form astrometric solver
  ┌──────────────────────────────────────────────────────────┐
  │  observations.csv  (>= 6 (timestamp, pixel, body) rows)  │
  │              │                                           │
  │              ▼                                           │
  │  celestial_calibration.solve_calibration()               │
  │              │                                           │
  │              ▼                                           │
  │  focal_length_px, yaw, pitch, roll, k1, k2  +  overlay   │
  └──────────────────────────────────────────────────────────┘
```

## Requirements
- Python 3.9+
- CUDA GPU recommended for Agent 3 (CPU fallback available, ~3× slower)
- ~2 GB disk space for the DepthPro model cache
- `scipy` and `skyfield` for the celestial calibration solver
  (see `celestial_calibration.py`); skyfield additionally downloads the
  `de421.bsp` JPL ephemeris (~17 MB) on first use.

## Installation

The repository is developed against a conda environment named `cg`. The
recommended workflow is:

```bash
git clone https://github.com/Ezharjan/AgenticWebCamCalibration.git
cd AgenticWebCamCalibration
conda activate cg                # all subsequent commands run in this env
pip install -r requirements.txt
```

Every script in this repo is intended to be run inside `conda activate cg`.

CUDA note: if pip installs a CPU-only PyTorch, reinstall the GPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

Run all three agents on the full dataset:
```bash
conda activate cg
python main.py --csv webcam_list.csv
```

Smoke test on 10 rows:
```bash
python main.py --csv webcam_list.csv --sample 10
```

Run only agents 1 and 2 (skip calibration):
```bash
python main.py --agents 1,2
```

Resume a previous run (already-processed rows are automatically skipped):
```bash
python main.py
```

Run the celestial calibration solver on the bundled example:
```bash
python celestial_calibration.py \
    --observations example_observations.csv \
    --image-width 1280 --image-height 720 \
    --image example_sky.png \
    --overlay output/example_overlay.png \
    --json-out output/example_calibration.json
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| --csv | webcam_list.csv | Input CSV file |
| --sample N | None | Process first N rows only |
| --agents 1,2,3 | 1,2,3 | Agents to run (comma-separated) |
| --skip-health | off | Skip Agent 1 |
| --skip-repair | off | Skip Agent 2 |
| --skip-calibration | off | Skip Agent 3 |

## Input CSV Format

The input CSV requires the following structure:

| Column | Description | Example |
|---|---|---|
| Name | Human-readable camera name | aeroclub cote d'or |
| ImageURL | Direct URL to the live JPEG/PNG image | `https://example.com/camera/image.jpg` |
| URLSource | Web page where the camera is embedded | `https://example.com/webcams` |
| Latitude | Camera latitude (decimal degrees) | `47.3841` |
| Longitude | Camera longitude (decimal degrees) | `4.9442` |
| RefreshRate | How often the image updates (informational) | `30 seconds` |

## Output

A **single CSV file**: `output/webcam_calibration_results.csv`

This file is the input CSV progressively enriched with columns from each agent.
Intermediate progress is saved periodically, so runs can be resumed.

### Columns Added by Agent 1 (Health Check)
| Column | Values | Notes |
|---|---|---|
| health_status | VALID / INVALID / EMPTY | EMPTY = blank/NaN ImageURL |
| health_failure_reason | dns_failure, timeout_connect, timeout_read, http_NNN, not_an_image, empty_response, connection_error, redirect_limit, unknown_error | "" if VALID |
| health_http_code | integer | 0 if connection never reached the server |
| health_check_timestamp | ISO 8601 UTC | |

### Columns Added by Agent 2 (URL Repair)
| Column | Values | Notes |
|---|---|---|
| original_image_url | URL string | Original ImageURL before repair (only set if repaired) |
| repair_status | REPAIRED / ALREADY_VALID / UNRESOLVED | |
| repair_method | direct_revalidation, strip_query, scrape, wayback, pattern_mutation, faa_api, faa_scrape | "" if not repaired |
| repair_attempts | integer | Number of strategies attempted |
| repair_timestamp | ISO 8601 UTC | |

### Columns Added by Agent 3 (Calibration)
| Column | Values | Notes |
|---|---|---|
| focal_length_px | float | Estimated focal length in pixels |
| fov_horizontal_deg | float | Horizontal field of view in degrees |
| fov_vertical_deg | float | Derived: $2 \arctan\!\bigl(\tan(FOV_h/2) / aspect\bigr)$ |
| image_width_px | int | |
| image_height_px | int | |
| focal_length_normalized | float | focal_length_px / image_width_px |
| sensor_aspect_ratio | float | width / height |
| calibration_confidence | HIGH / MEDIUM / LOW | Based on FOV plausibility |
| calibration_status | SUCCESS / FAILED / SKIPPED | SKIPPED = no valid URL |
| calibration_failure_reason | download_error, image_too_small, image_quality_*, inference_error, model_unavailable, no_valid_url | "" if SUCCESS |
| calibration_timestamp | ISO 8601 UTC | |

## Agentic Behaviour

Each agent uses autonomous decision-making:

**Agent 1** retries transient failures (timeouts, connection errors) with
escalating timeouts before marking a URL as INVALID.

**Agent 2** selects repair strategies based on the failure type reported by
Agent 1. For timeouts it re-validates with a longer timeout; for HTTP errors
it tries source scraping and Wayback; for FAA URLs it uses the FAA-specific
API. Candidates are scored and the highest-quality match is selected.

**Agent 3** assesses image quality before running inference (rejects solid-colour
placeholders, overexposed, or too-dark frames). If the initial FOV estimate is
physically implausible, it retries with a center-cropped and then a resized
version. Results are assigned a confidence level (HIGH / MEDIUM / LOW) based on
whether the FOV falls within the typical webcam range (30°–130°).

## Free Resources Used

| Resource | Purpose | Cost |
|---|---|---|
| requests + BeautifulSoup4 | HTML scraping for URL repair | Free |
| Wayback Machine CDX API | Last-known-good snapshot lookup | Free, no key |
| apple/DepthPro-hf (Hugging Face) | Camera intrinsic estimation | Free, runs locally |
| PyTorch | Model inference backend | Free |
| scipy, skyfield, numpy | Celestial calibration solver | Free |
| JPL DE421 ephemeris | Sun/moon/planet positions for skyfield | Free, public domain |

No API keys are required.

## Troubleshooting

**DepthPro model download is slow**
The model is ~1.4 GB and downloaded once to `~/.cache/huggingface/hub`.
Ensure sufficient disk space and a stable connection.

**CUDA out-of-memory**
Agent 3 attempts CPU fallback. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
in your environment for small GPUs.

**Agent 2 is slow on large datasets**
URL scraping respects a 2-second per-domain rate limit. Use `--sample 100`
to validate behaviour first.

**Resuming after interruption**
Simply re-run `python main.py`. Rows already processed by each agent are
detected automatically (by checking status columns) and skipped.

## Celestial Calibration Solver (`celestial_calibration.py`)

In addition to Agent 3's monocular DepthPro estimator, the project ships a
**closed-form astrometric solver** that recovers full camera geometry from
observations of celestial bodies. Given **>= 6** triplets of
`(timestamp_utc, pixel_xy, body)`, it solves for:

| Parameter | Meaning |
|---|---|
| `focal_length_px` | pinhole focal length (px), square pixels |
| `yaw_deg`         | compass heading of the optical axis (positive: North -> East) |
| `pitch_deg`       | tilt of the optical axis (positive: above horizon) |
| `roll_deg`        | rotation about the optical axis (positive: image content CCW) |
| `k1`, `k2`        | Brown-Conrady radial distortion coefficients |

### Camera and world model

The solver assumes a pinhole camera with single focal length `f`, principal
point fixed at the image centre (configurable via `--cx --cy`), and 2-term
radial distortion. The camera frame is right-handed with `+x` right, `+y`
down, `+z` along the optical axis. The world is local **ENU**
(East-North-Up) at the camera location. At `yaw=pitch=roll=0` the optical
axis points North and `+x` points East.

For a celestial body whose true direction in ENU is the unit vector
`v = (sin(Az) cos(El), cos(Az) cos(El), sin(El))`, the forward projection is:

```
v_cam   = R_roll(roll) * R_yaw_pitch(yaw, pitch) * v
x_n     = v_cam[0] / v_cam[2]
y_n     = v_cam[1] / v_cam[2]
r2      = x_n^2 + y_n^2
(x_d, y_d) = (1 + k1*r2 + k2*r2^2) * (x_n, y_n)
(u, v_pix) = (f * x_d + cx, f * y_d + cy)
```

The solver minimises the sum of squared pixel residuals
`sum_i ||(u_i, v_i) - (u_obs_i, v_obs_i)||^2` using
`scipy.optimize.least_squares` (Trust-Region/Levenberg-Marquardt with
finite-difference Jacobian). A pure-numpy LM fallback is included so the
solver can also run without scipy (slower; recommended only for testing).

### Observation CSV format

```csv
timestamp_utc,pixel_x,pixel_y,body,az_deg,el_deg,weight
2026-01-15T03:00:00Z,208.36,368.39,sirius,,,
2026-01-15T03:15:00Z,676.80,107.33,betelgeuse,,,
...
```

- `timestamp_utc` — ISO-8601 (e.g. `2026-01-15T03:00:00Z`). Required.
- `pixel_x`, `pixel_y` — observed pixel of the body in the calibration image.
- `body` — `sun`, `moon`, `mercury`, `venus`, `mars`, `jupiter`, `saturn`,
  `uranus`, `neptune`, or one of the named stars (`polaris`, `sirius`,
  `vega`, `arcturus`, `capella`, `rigel`, `procyon`, `betelgeuse`,
  `altair`, `aldebaran`, `antares`, `spica`, `pollux`, `deneb`,
  `regulus`, `fomalhaut`, `mira`, `canopus`).
- `az_deg`, `el_deg` (optional) — apparent compass azimuth and elevation in
  degrees. If supplied, the solver uses these directly and skips the skyfield
  lookup (useful when you already have observed positions or for offline use).
- `weight` (optional, default `1.0`) — per-observation residual weight.

### Quick start

```bash
conda activate cg                          # use the project env
python celestial_calibration.py \
    --observations example_observations.csv \
    --image-width 1280 --image-height 720 \
    --image example_sky.png \
    --overlay output/example_overlay.png \
    --json-out output/example_calibration.json
```

If your CSV omits `az_deg`/`el_deg`, also pass:

```bash
    --latitude 37.7749 --longitude -122.4194 --elevation 30
```

so skyfield can compute apparent body positions from the camera location.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--observations` | required | Path to the input CSV |
| `--image-width`  | required | Image width in pixels |
| `--image-height` | required | Image height in pixels |
| `--latitude`     | none | Camera latitude (deg). Required unless every row has explicit `az_deg`/`el_deg`. |
| `--longitude`    | none | Camera longitude (deg). |
| `--elevation`    | 0    | Camera elevation (m, WGS84). |
| `--cx`, `--cy`   | image centre | Override the principal point. |
| `--fix-distortion` | off | Solve with `k1=k2=0` (4-parameter solve). |
| `--fix-roll`       | off | Solve with `roll=0` (5-parameter solve, or 3 with `--fix-distortion`). |
| `--image`        | none | Calibration image path. Required for overlay. |
| `--overlay`      | none | Where to save the validation overlay PNG. |
| `--json-out`     | none | Where to save the full result JSON. |
| `--verbose`      | 0    | scipy `least_squares` verbose level (0/1/2). |

### Output

The solver prints a result table and (when `--json-out` is given) writes a
JSON file with the full result, including per-observation reprojection
residuals. Example fields:

```json
{
  "focal_length_px": 949.52, "yaw_deg": 130.00, "pitch_deg": 10.00,
  "roll_deg": -1.48, "k1": -0.0677, "k2": 0.0179,
  "image_width": 1280, "image_height": 720, "cx": 640.0, "cy": 360.0,
  "rms_reprojection_error_px": 0.54, "max_reprojection_error_px": 0.74,
  "median_reprojection_error_px": 0.48,
  "n_observations": 12, "converged": true,
  "fov_horizontal_deg": 67.96, "fov_vertical_deg": 41.53
}
```

### Validation overlay

When `--image` and `--overlay` are supplied, the solver renders the input
image with:

- **Green circles** at the observed pixel positions, labelled with the
  body name;
- **Red crosses** at the predicted pixel positions (after applying the
  solved geometry and distortion);
- **Yellow segments** connecting each observed/predicted pair (the
  reprojection residual).

A header band reports the recovered `f`, FOV, `yaw/pitch/roll`, distortion,
RMS / max reprojection error and observation count. Visually, when
calibration is correct the green circles and red crosses overlap to within
a pixel.

### OpenCV-compatible export

Calibration results can be exported to standard interchange formats so the
camera can be used directly by `cv2`, ROS, COLMAP-style pipelines, or any
tool that consumes a camera matrix + distortion coefficients. Our distortion
model is exactly OpenCV's pinhole + radial-tangential model with
`p1 = p2 = k3 = 0`, so the exported `dist_coeffs` is the canonical 5-vector
`[k1, k2, 0, 0, 0]`.

#### CLI flags

| Flag | Format | Notes |
|---|---|---|
| `--opencv-yaml PATH` | `%YAML:1.0` | Loadable by `cv2.FileStorage`. |
| `--opencv-json PATH` | JSON | Same schema as cv2's JSON FileStorage output. |
| `--opencv-npz  PATH` | NumPy `.npz` | `camera_matrix`, `dist_coeffs`, `image_size`, `rvec`, `tvec`, `rotation_matrix`. |
| `--ros-camera-info PATH` | ROS sensor_msgs/CameraInfo YAML | `plumb_bob` model; includes rectification + projection matrices. |

Example:

```bash
conda activate cg
python celestial_calibration.py \
    --observations example_observations.csv \
    --image-width 1280 --image-height 720 \
    --opencv-yaml output/example_calibration.yaml \
    --opencv-json output/example_calibration_opencv.json \
    --opencv-npz  output/example_calibration.npz \
    --ros-camera-info output/example_camera_info.yaml
```

#### Drop-in OpenCV usage

After exporting, `cv2` can use the calibration directly without any glue
code:

```python
import cv2

# YAML
fs = cv2.FileStorage("output/example_calibration.yaml", cv2.FILE_STORAGE_READ)
K     = fs.getNode("camera_matrix").mat()
dist  = fs.getNode("distortion_coefficients").mat()
rvec  = fs.getNode("rotation_vector").mat()
tvec  = fs.getNode("translation_vector").mat()
fs.release()

# Undistort an image
img        = cv2.imread("frame.jpg")
undistorted = cv2.undistort(img, K, dist)

# Project a celestial direction back to pixel space
direction_enu = ...  # shape (N, 1, 3) unit vectors
pixels, _ = cv2.projectPoints(direction_enu, rvec, tvec, K, dist)

# Or build undistortion maps for video-rate use
mapx, mapy = cv2.initUndistortRectifyMap(
    K, dist, None, K, (1280, 720), cv2.CV_32FC1)
undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
```

NumPy `.npz` is equally easy:

```python
import numpy as np
d = np.load("output/example_calibration.npz")
K, dist, rvec, tvec = d["camera_matrix"], d["dist_coeffs"], d["rvec"], d["tvec"]
W, H = d["image_size"]
```

#### Programmatic export

```python
from celestial_calibration import solve_calibration
from opencv_export import (
    to_opencv_intrinsics,
    export_opencv_yaml, export_opencv_json,
    export_numpy_npz, export_ros_camera_info,
)

result = solve_calibration(observations, image_width=W, image_height=H)
cv = to_opencv_intrinsics(result)
print("K =", cv["camera_matrix"])
print("dist =", cv["dist_coeffs"])
print("rvec =", cv["rvec"], "tvec =", cv["tvec"])

export_opencv_yaml(result, "calibration.yaml")
export_opencv_json(result, "calibration.json")
export_numpy_npz (result, "calibration.npz")
export_ros_camera_info(result, "camera_info.yaml")
```

#### Equivalence guarantee

`test_opencv_export.py` includes a strict equivalence test that re-projects
celestial directions through the exported `(K, dist, rvec, tvec)` using
both a pure-NumPy implementation and (when installed) the real
`cv2.projectPoints`. Both must agree with the native solver to
floating-point precision. The current pass shows max disagreement
~2.3e-13 px against the pure-NumPy OpenCV-style model and ~2.7e-7 px
through `cv2.FileStorage + cv2.projectPoints` (round-tripped through
ASCII floats).

#### Notes on `rvec`/`tvec`

- `rvec` is the OpenCV Rodrigues form of `R_world_to_cam` (the rotation
  from local ENU directions to camera coordinates).
- `tvec` is `(0, 0, 0)`: the camera location is the origin of the local
  ENU frame, and celestial bodies are unit-vector directions at infinity.
  If you want to project a 3D world point in some other frame, transform
  it into ENU first (or compose with your own world-to-ENU transform).

### Programmatic API

```python
from datetime import datetime, timezone
from celestial_calibration import (
    Observation, solve_calibration, render_validation_overlay,
)

obs = [
    Observation(timestamp=datetime(2026, 1, 15, 3, 0, tzinfo=timezone.utc),
                pixel_x=208.36, pixel_y=368.39, body="sirius",
                az_deg=105.0, el_deg=8.0),
    # ... at least 6 ...
]
res = solve_calibration(obs, image_width=1280, image_height=720)
print(res.focal_length_px, res.yaw_deg, res.pitch_deg, res.roll_deg,
      res.k1, res.k2, res.rms_reprojection_error_px)
render_validation_overlay("sky.png", res, "overlay.png")
```

### Self-tests

```bash
conda activate cg
python test_celestial_calibration.py        # solver
python test_opencv_export.py                # OpenCV-export round-trips
```

The first script checks the forward model against hand-computed cases
(zero-rotation / yaw / pitch / focal scaling / radial distortion), verifies
that residuals are zero at the ground-truth parameters, and runs a
synthetic round-trip:

| Scenario | Expected behaviour |
|---|---|
| Noise-free, 25 obs spread across FOV  | All 6 parameters recovered to numerical precision; RMS <= 1e-3 px. |
| Gaussian noise sigma = 0.3 px, 25 obs | RMS approx noise floor; `f` within a few px, angles within <= 0.05 deg, `k1`/`k2` within <= 0.02. |

### Practical tips

- **Spread observations across the field of view.** If all observations
  cluster near the optical axis, `f` and `k1` become highly correlated and
  the distortion coefficients will be poorly determined. For a full 6-DOF
  fit, aim for observations at the corners as well.
- **At least 6 observations are required** (8+ recommended) — there are 6
  free parameters.
- For very narrow-FOV cameras or short observation sessions, use
  `--fix-distortion` to solve only for `f, yaw, pitch, roll`.
- Marker localisation noise of ~0.3 px (typical for a centroided sun or
  bright star) yields ~5 px focal-length uncertainty over a typical webcam
  FOV. Sub-arcminute angular accuracy is achievable.
- The principal point `(cx, cy)` defaults to the image centre. Most
  consumer webcams are well-modelled this way; if you have a known offset,
  pass `--cx --cy`.

## Project Structure

```text
AgenticWebCamCalibration/
├── main.py                       Orchestrator & CLI entry point
├── agent1_health_check.py        URL validation with agentic retry
├── agent2_url_repair.py          Multi-strategy agentic URL repair
├── agent3_calibration.py         DepthPro calibration with self-correction
├── celestial_calibration.py      Astrometric solver (focal/yaw/pitch/roll/k1/k2)
├── opencv_export.py              OpenCV/ROS/NumPy calibration exporters
├── test_celestial_calibration.py Self-tests for the solver
├── test_opencv_export.py         Self-tests for the OpenCV export
├── example_observations.csv      Sample input for the celestial solver
├── example_sky.png               Synthetic image matching the example obs
├── utils.py                      Shared helpers
├── requirements.txt              Dependencies
├── README.md                     This file
└── output/                       CSV / JSON / overlay outputs (auto-created)
```

## License
MIT License — see LICENSE file.
