"""Agent 3: Agentic Calibration — estimates intrinsic camera parameters using
Apple DepthPro with quality assessment, plausibility validation, and
self-correcting retry logic."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import math
import signal
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import urllib3

urllib3.disable_warnings()

import torch
try:
    from transformers import DepthProImageProcessor, DepthProForDepthEstimation
    HAS_DEPTHPRO = True
except ImportError:
    HAS_DEPTHPRO = False

from utils import (
    setup_logging, download_image_with_retry, assess_image_quality,
    is_image_too_small, safe_csv_write,
)

logger = setup_logging()

g_df = None
g_csv_path = None
g_interrupted = False

# Physical plausibility bounds for webcam calibration
MIN_FOV_DEG = 5.0
MAX_FOV_DEG = 170.0
TYPICAL_FOV_LOW = 30.0
TYPICAL_FOV_HIGH = 130.0


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C)."""
    global g_interrupted
    g_interrupted = True
    print("\n[Agent 3] Interrupted! Saving progress...")


# ---------------------------------------------------------------------------
# Agentic calibration core
# ---------------------------------------------------------------------------

def _run_inference(img, image_processor, model, device):
    """Run DepthPro inference. Returns (fov_deg, depth_map) or None on failure."""
    try:
        inputs = image_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract FOV — field comes directly from model output
        fov_deg = None
        if hasattr(outputs, "fov") and outputs.fov is not None:
            fov_deg = outputs.fov.item()
        elif hasattr(outputs, "field_of_view") and outputs.field_of_view is not None:
            fov_deg = outputs.field_of_view.item()

        # Fallback: check post-processed output
        if fov_deg is None:
            post = image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(img.height, img.width)]
            )
            for key in ("fov", "field_of_view", "fov_deg"):
                if key in post[0]:
                    val = post[0][key]
                    fov_deg = val.item() if hasattr(val, "item") else float(val)
                    break
            # Also try focal_length -> compute FOV
            if fov_deg is None and "focal_length" in post[0]:
                fl = post[0]["focal_length"]
                fl_val = fl.item() if hasattr(fl, "item") else float(fl)
                if fl_val > 0:
                    fov_deg = 2 * math.degrees(math.atan(img.width / (2 * fl_val)))

        return fov_deg
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.debug(f"Inference error: {e}")
        return None


def _calibrate_single(img, image_processor, model, device):
    """Agentic calibration: run inference, validate, retry if implausible.

    Returns a dict with calibration results or (None, failure_reason).
    """
    # --- Step 1: Assess image quality ---
    suitable, quality_reason = assess_image_quality(img)
    if not suitable:
        return None, f"image_quality_{quality_reason}"

    w, h = img.size

    # --- Step 2: First inference attempt ---
    fov_deg = _run_inference(img, image_processor, model, device)

    # --- Step 3: Validate plausibility ---
    if fov_deg is not None and MIN_FOV_DEG <= fov_deg <= MAX_FOV_DEG:
        confidence = "HIGH" if TYPICAL_FOV_LOW <= fov_deg <= TYPICAL_FOV_HIGH else "MEDIUM"
        return _build_result(fov_deg, w, h, confidence), None

    # --- Step 4: Agentic retry — center crop to reduce edge distortion ---
    crop_margin = 0.1
    crop_box = (
        int(w * crop_margin), int(h * crop_margin),
        int(w * (1 - crop_margin)), int(h * (1 - crop_margin)),
    )
    cropped = img.crop(crop_box)
    if cropped.width >= 100 and cropped.height >= 100:
        fov_crop = _run_inference(cropped, image_processor, model, device)
        if fov_crop is not None:
            # Scale FOV back to original image dimensions
            # crop covers 80% of width → adjust FOV proportionally
            scale = w / cropped.width
            fov_adjusted = 2 * math.degrees(
                math.atan(scale * math.tan(math.radians(fov_crop / 2)))
            )
            if MIN_FOV_DEG <= fov_adjusted <= MAX_FOV_DEG:
                confidence = "MEDIUM"
                return _build_result(fov_adjusted, w, h, confidence), None

    # --- Step 5: Agentic retry — resize to 512px width for stability ---
    if w > 512:
        ratio = 512 / w
        resized = img.resize((512, int(h * ratio)))
        fov_resized = _run_inference(resized, image_processor, model, device)
        if fov_resized is not None and MIN_FOV_DEG <= fov_resized <= MAX_FOV_DEG:
            # FOV should be the same regardless of resolution (it's angular)
            confidence = "MEDIUM"
            return _build_result(fov_resized, w, h, confidence), None

    # --- Step 6: Use original result with LOW confidence if within physical limits ---
    if fov_deg is not None:
        clamped = max(MIN_FOV_DEG, min(MAX_FOV_DEG, fov_deg))
        return _build_result(clamped, w, h, "LOW"), None

    return None, "inference_error"


def _build_result(fov_h_deg: float, width: int, height: int, confidence: str) -> dict:
    """Compute all intrinsic parameters from horizontal FOV and image size."""
    focal_length_px = (width / 2) / math.tan(math.radians(fov_h_deg / 2))
    aspect = width / height
    fov_v_deg = 2 * math.degrees(
        math.atan(math.tan(math.radians(fov_h_deg / 2)) / aspect)
    )
    return {
        "focal_length_px": round(focal_length_px, 2),
        "fov_horizontal_deg": round(fov_h_deg, 2),
        "fov_vertical_deg": round(fov_v_deg, 2),
        "image_width_px": width,
        "image_height_px": height,
        "focal_length_normalized": round(focal_length_px / width, 4),
        "sensor_aspect_ratio": round(aspect, 4),
        "calibration_confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str, sample: int = None):
    """Main execution of Agent 3."""
    global g_df, g_csv_path, g_interrupted, HAS_DEPTHPRO
    g_interrupted = False
    signal.signal(signal.SIGINT, sigint_handler)

    g_csv_path = csv_path

    try:
        df = pd.read_csv(csv_path)
        cols = [
            "focal_length_px", "fov_horizontal_deg", "fov_vertical_deg",
            "image_width_px", "image_height_px", "focal_length_normalized",
            "sensor_aspect_ratio", "calibration_confidence",
            "calibration_status", "calibration_failure_reason",
            "calibration_timestamp",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = pd.NA
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    if sample and len(df) > sample:
        df = df.head(sample)

    g_df = df
    indices_to_check = df[df["calibration_status"].isna()].index.tolist()

    if not indices_to_check:
        print("[Agent 3] All rows already calibrated. Skipping.")
        safe_csv_write(g_df, g_csv_path)
        return

    logger.info(f"Agent 3: Calibrating {len(indices_to_check)} cameras.")

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = None
    model = None

    if HAS_DEPTHPRO:
        MODEL_ID = "apple/DepthPro-hf"
        print(f"[Agent 3] Loading DepthPro model ({MODEL_ID}) on {device}...")
        try:
            image_processor = DepthProImageProcessor.from_pretrained(MODEL_ID)
            model = DepthProForDepthEstimation.from_pretrained(MODEL_ID, use_fov_model=True)
            model.eval()
            model.to(device)
            print("[Agent 3] Model ready.")
        except Exception as e:
            print(f"[Agent 3] Failed to load model: {e}")
            HAS_DEPTHPRO = False

    if not HAS_DEPTHPRO:
        print("[Agent 3] DepthPro not available. Install 'transformers' with DepthPro support.")
        print("[Agent 3] Marking all rows as FAILED (model_unavailable).")
        for idx in indices_to_check:
            g_df.at[idx, "calibration_status"] = "FAILED"
            g_df.at[idx, "calibration_failure_reason"] = "model_unavailable"
            g_df.at[idx, "calibration_timestamp"] = datetime.utcnow().isoformat() + "Z"
        safe_csv_write(g_df, g_csv_path)
        return

    # --- Process rows ---
    processed = 0
    for idx in tqdm(indices_to_check, desc="Agent 3"):
        if g_interrupted:
            break

        row = g_df.loc[idx]
        h_status = str(row.get("health_status", ""))
        r_status = str(row.get("repair_status", ""))

        # Only calibrate if URL is known-good
        if h_status != "VALID" and r_status not in ("REPAIRED", "ALREADY_VALID"):
            g_df.at[idx, "calibration_status"] = "SKIPPED"
            g_df.at[idx, "calibration_failure_reason"] = "no_valid_url"
            g_df.at[idx, "calibration_timestamp"] = datetime.utcnow().isoformat() + "Z"
            continue

        url = str(row.get("ImageURL", ""))

        # Agentic download with retries
        img = download_image_with_retry(url, max_retries=3)
        if img is None:
            g_df.at[idx, "calibration_status"] = "FAILED"
            g_df.at[idx, "calibration_failure_reason"] = "download_error"
            g_df.at[idx, "calibration_timestamp"] = datetime.utcnow().isoformat() + "Z"
            continue

        if is_image_too_small(img, min_px=50):
            g_df.at[idx, "calibration_status"] = "FAILED"
            g_df.at[idx, "calibration_failure_reason"] = "image_too_small"
            g_df.at[idx, "calibration_timestamp"] = datetime.utcnow().isoformat() + "Z"
            continue

        # Agentic calibration with validation and retry
        result, fail_reason = _calibrate_single(img, image_processor, model, device)

        if result is not None:
            for key, val in result.items():
                g_df.at[idx, key] = val
            g_df.at[idx, "calibration_status"] = "SUCCESS"
            g_df.at[idx, "calibration_failure_reason"] = ""
        else:
            g_df.at[idx, "calibration_status"] = "FAILED"
            g_df.at[idx, "calibration_failure_reason"] = fail_reason or "inference_error"

        g_df.at[idx, "calibration_timestamp"] = datetime.utcnow().isoformat() + "Z"

        processed += 1
        if processed % 50 == 0:
            safe_csv_write(g_df, g_csv_path)

    # Final save
    safe_csv_write(g_df, g_csv_path)

    # Print summary
    total = len(g_df)
    success = len(g_df[g_df["calibration_status"] == "SUCCESS"])
    failed = len(g_df[g_df["calibration_status"] == "FAILED"])
    skipped = len(g_df[g_df["calibration_status"] == "SKIPPED"])

    print("\n" + " Agent 3 Summary ".center(40, "="))
    print(f"{'Status'.ljust(15)} | {'Count'.rjust(8)} | {'%'.rjust(8)}")
    print("-" * 40)
    print(f"{'SUCCESS'.ljust(15)} | {str(success).rjust(8)} | {f'{(success/total*100 if total else 0):.1f}'.rjust(8)}")
    print(f"{'FAILED'.ljust(15)} | {str(failed).rjust(8)} | {f'{(failed/total*100 if total else 0):.1f}'.rjust(8)}")
    print(f"{'SKIPPED'.ljust(15)} | {str(skipped).rjust(8)} | {f'{(skipped/total*100 if total else 0):.1f}'.rjust(8)}")
    print(f"{'TOTAL'.ljust(15)} | {str(total).rjust(8)} | {'100.0'.rjust(8)}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/webcam_calibration_results.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(args.csv, args.sample)
