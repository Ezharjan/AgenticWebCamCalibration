"""Orchestrator — runs the three-agent pipeline and produces a single output CSV."""
import sys
import shutil
import argparse
import time
import os

import pandas as pd

from agent1_health_check import main as agent1_main
from agent2_url_repair import main as agent2_main
from agent3_calibration import main as agent3_main

OUTPUT_CSV = "output/webcam_calibration_results.csv"


def main():
    parser = argparse.ArgumentParser(description="Agentic Webcam Calibration Tool")
    parser.add_argument("--csv", default="webcam_list.csv", help="Input CSV file path")
    parser.add_argument("--sample", type=int, default=None, help="Process first N rows")
    parser.add_argument("--agents", default="1,2,3", help="Comma-separated agent list")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    agent_str = args.agents.split(",")
    agents_to_run = set(int(a.strip()) for a in agent_str if a.strip().isdigit())
    if args.skip_health:
        agents_to_run.discard(1)
    if args.skip_repair:
        agents_to_run.discard(2)
    if args.skip_calibration:
        agents_to_run.discard(3)

    os.makedirs("output", exist_ok=True)

    # Initialise the single output CSV from the input if it doesn't exist yet
    if not os.path.exists(OUTPUT_CSV):
        if not os.path.exists(args.csv):
            print(f"[Orchestrator] Input file not found: {args.csv}")
            sys.exit(1)
        shutil.copy2(args.csv, OUTPUT_CSV)
        print(f"[Orchestrator] Initialised {OUTPUT_CSV} from {args.csv}")
    else:
        print(f"[Orchestrator] Resuming from existing {OUTPUT_CSV}")

    timers = {1: 0.0, 2: 0.0, 3: 0.0}

    try:
        if 1 in agents_to_run:
            print("\n[Orchestrator] Starting Agent 1: Health Check")
            t0 = time.perf_counter()
            agent1_main(OUTPUT_CSV, args.sample)
            timers[1] = time.perf_counter() - t0

        if 2 in agents_to_run:
            print("\n[Orchestrator] Starting Agent 2: URL Repair")
            t0 = time.perf_counter()
            agent2_main(OUTPUT_CSV, args.sample)
            timers[2] = time.perf_counter() - t0

        if 3 in agents_to_run:
            print("\n[Orchestrator] Starting Agent 3: Calibration")
            t0 = time.perf_counter()
            agent3_main(OUTPUT_CSV, args.sample)
            timers[3] = time.perf_counter() - t0

    except Exception as e:
        print(f"\n[Orchestrator] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    # --- Final summary ---
    try:
        df = pd.read_csv(OUTPUT_CSV)
        total = len(df)
    except Exception:
        total = 0

    if total == 0:
        print("\n[Orchestrator] No rows processed.")
        sys.exit(1)

    valid = len(df[df["health_status"] == "VALID"]) if "health_status" in df.columns else 0
    repaired = len(df[df["repair_status"] == "REPAIRED"]) if "repair_status" in df.columns else 0
    already_valid = len(df[df["repair_status"] == "ALREADY_VALID"]) if "repair_status" in df.columns else 0
    calibrated = len(df[df["calibration_status"] == "SUCCESS"]) if "calibration_status" in df.columns else 0
    cal_failed = len(df[df["calibration_status"] == "FAILED"]) if "calibration_status" in df.columns else 0
    cal_skipped = len(df[df["calibration_status"] == "SKIPPED"]) if "calibration_status" in df.columns else 0

    pct = lambda n: f"{n / total * 100:.1f}"

    print("\n" + "=" * 60)
    print(" FINAL PIPELINE SUMMARY ".center(60))
    print("=" * 60)
    print(f"  Output file : {OUTPUT_CSV}")
    print(f"  Total rows  : {total}")
    print()
    print(f"  {'Metric':<28} {'Count':>7} {'%':>7} {'Time':>8}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*8}")
    print(f"  {'Health: VALID':<28} {valid:>7} {pct(valid):>7} {timers[1]:>7.1f}s")
    print(f"  {'Repair: ALREADY_VALID':<28} {already_valid:>7} {pct(already_valid):>7}")
    print(f"  {'Repair: REPAIRED':<28} {repaired:>7} {pct(repaired):>7} {timers[2]:>7.1f}s")
    print(f"  {'Calibration: SUCCESS':<28} {calibrated:>7} {pct(calibrated):>7} {timers[3]:>7.1f}s")
    print(f"  {'Calibration: FAILED':<28} {cal_failed:>7} {pct(cal_failed):>7}")
    print(f"  {'Calibration: SKIPPED':<28} {cal_skipped:>7} {pct(cal_skipped):>7}")
    print("=" * 60)


if __name__ == "__main__":
    main()
