"""Agent 1: Health Check — validates every ImageURL with agentic retry logic."""
import os
import sys
import signal
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

import urllib3
urllib3.disable_warnings()

from utils import setup_logging, validate_image_url, safe_csv_write

logger = setup_logging()

g_df = None
g_csv_path = None
g_interrupted = False

MAX_RETRIES = 2  # Agentic: retry transient failures with longer timeouts

TRANSIENT_ERRORS = frozenset({
    "timeout_connect", "timeout_read", "connection_error", "unknown_error",
})


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) to flush data and exit cleanly."""
    global g_interrupted
    g_interrupted = True
    print("\n[Agent 1] Interrupted! Saving progress...")


def process_row(args):
    """Validate a single URL with agentic retry on transient errors."""
    index, row = args

    if pd.notna(row.get("health_status")):
        return None  # Already processed

    url = row.get("ImageURL", "")

    # Agentic retry loop — escalate timeout on transient failures
    last_valid, last_reason, last_code = False, "unknown_error", 0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            timeout = (10 * attempt, 15 * attempt)
            valid, reason, status_code = validate_image_url(url, timeout=timeout)
            last_valid, last_reason, last_code = valid, reason, status_code

            if valid:
                break
            # Only retry on transient errors
            if reason not in TRANSIENT_ERRORS:
                break
        except Exception:
            last_valid, last_reason, last_code = False, "unknown_error", 0

    h_status = "VALID" if last_valid else ("EMPTY" if last_reason == "empty" else "INVALID")
    h_reason = "" if last_valid else last_reason
    return (index, h_status, h_reason, last_code)


def main(csv_path: str, sample: int = None):
    """Main execution of Agent 1."""
    global g_df, g_csv_path, g_interrupted
    g_interrupted = False
    signal.signal(signal.SIGINT, sigint_handler)

    g_csv_path = csv_path

    try:
        df = pd.read_csv(csv_path)
        for col in ["health_status", "health_failure_reason", "health_http_code",
                     "health_check_timestamp"]:
            if col not in df.columns:
                df[col] = pd.NA
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    if sample and len(df) > sample:
        df = df.head(sample)

    g_df = df
    indices_to_check = df[df["health_status"].isna()].index.tolist()

    if not indices_to_check:
        print("[Agent 1] All rows already checked. Skipping.")
        safe_csv_write(g_df, g_csv_path)
        return

    logger.info(f"Agent 1: Checking {len(indices_to_check)} URLs.")

    processed = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        tasks = [(idx, g_df.loc[idx]) for idx in indices_to_check]
        future_to_idx = {executor.submit(process_row, t): t[0] for t in tasks}

        for future in tqdm(as_completed(future_to_idx), total=len(indices_to_check),
                           desc="Agent 1"):
            if g_interrupted:
                break

            idx = future_to_idx[future]
            try:
                res = future.result()
                if res is not None:
                    _, h_status, h_reason, h_code = res
                    g_df.at[idx, "health_status"] = h_status
                    g_df.at[idx, "health_failure_reason"] = h_reason
                    g_df.at[idx, "health_http_code"] = h_code if h_code else pd.NA
                    g_df.at[idx, "health_check_timestamp"] = datetime.utcnow().isoformat() + "Z"
            except Exception as e:
                logger.error("Error processing row", extra={"row": idx, "detail": str(e)})
                g_df.at[idx, "health_status"] = "INVALID"
                g_df.at[idx, "health_failure_reason"] = "processing_error"
                g_df.at[idx, "health_check_timestamp"] = datetime.utcnow().isoformat() + "Z"

            processed += 1
            if processed % 200 == 0:
                safe_csv_write(g_df, g_csv_path)

    # Final save
    safe_csv_write(g_df, g_csv_path)

    # Print summary
    total = len(g_df)
    valid_count = len(g_df[g_df["health_status"] == "VALID"])
    invalid_count = len(g_df[g_df["health_status"] == "INVALID"])
    empty_count = len(g_df[g_df["health_status"] == "EMPTY"])

    print("\n" + " Agent 1 Summary ".center(40, "="))
    print(f"{'Status'.ljust(15)} | {'Count'.rjust(8)} | {'%'.rjust(8)}")
    print("-" * 40)
    print(f"{'VALID'.ljust(15)} | {str(valid_count).rjust(8)} | {f'{(valid_count/total*100):.1f}'.rjust(8)}")
    print(f"{'INVALID'.ljust(15)} | {str(invalid_count).rjust(8)} | {f'{(invalid_count/total*100):.1f}'.rjust(8)}")
    print(f"{'EMPTY'.ljust(15)} | {str(empty_count).rjust(8)} | {f'{(empty_count/total*100):.1f}'.rjust(8)}")
    print(f"{'TOTAL'.ljust(15)} | {str(total).rjust(8)} | {'100.0'.rjust(8)}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="webcam_list.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(args.csv, args.sample)
