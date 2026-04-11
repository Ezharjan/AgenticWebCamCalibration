# Agentic Webcam Calibration Tool

## Overview
A three-agent pipeline that validates, repairs, and calibrates live webcam URLs
using Apple DepthPro — entirely free and local, with no paid APIs or keys.

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
```

## Requirements
- Python 3.9+
- CUDA GPU recommended for Agent 3 (CPU fallback available, ~3× slower)
- ~2 GB disk space for the DepthPro model cache

## Installation

```bash
git clone https://github.com/Ezharjan/AgenticWebCamCalibration.git
cd AgenticWebCamCalibration
pip install -r requirements.txt
```

CUDA note: if pip installs a CPU-only PyTorch, reinstall the GPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

Run all three agents on the full dataset:
```bash
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

## Project Structure

```text
AgenticWebCamCalibration/
├── main.py                  Orchestrator & CLI entry point
├── agent1_health_check.py   URL validation with agentic retry
├── agent2_url_repair.py     Multi-strategy agentic URL repair
├── agent3_calibration.py    DepthPro calibration with self-correction
├── utils.py                 Shared helpers (validation, rate limiter, logging)
├── requirements.txt         Dependencies
├── README.md                This file
└── output/                  Single output CSV (auto-created)
```

## License
MIT License — see LICENSE file.
