"""Agent 2: URL Repair — agentic multi-strategy repair for broken ImageURLs."""
import os
import sys
import time
import signal
import re
import urllib.parse
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

import urllib3
import requests
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

urllib3.disable_warnings()

from utils import (
    setup_logging, validate_image_url, normalize_url, get_domain,
    RateLimiter, safe_csv_write,
)

logger = setup_logging()

g_df = None
g_csv_path = None
g_interrupted = False
rate_limiter = RateLimiter()


def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C)."""
    global g_interrupted
    g_interrupted = True
    print("\n[Agent 2] Interrupted! Saving progress...")


# ---------------------------------------------------------------------------
# HTML scraping helpers
# ---------------------------------------------------------------------------

def find_urls_in_html(html: str, base_url: str = "") -> list:
    """Extract candidate image URLs from HTML, resolving relative paths."""
    candidates = []
    if not HAS_BS4:
        # Fallback: regex only
        raw = re.findall(
            r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp|bmp)(?:\?[^\s"\'<>]*)?',
            html,
        )
        return list(set(raw))

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # <img> src and data-src
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original"):
            val = img.get(attr)
            if val:
                candidates.append(val)

    # <source> srcset
    for source in soup.find_all("source"):
        srcset = source.get("srcset")
        if srcset:
            for part in srcset.split(","):
                part_url = part.strip().split(" ")[0]
                if part_url:
                    candidates.append(part_url)

    # <a href> pointing to image files
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href", "")
        if href and href.lower().split("?")[0].endswith(
            (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
        ):
            candidates.append(href)

    # Regex fallback for URLs embedded in JS / inline styles
    raw = re.findall(
        r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp|bmp)(?:\?[^\s"\'<>]*)?',
        html,
    )
    candidates.extend(raw)

    # Resolve relative URLs
    resolved = []
    for c in candidates:
        if c.startswith("//"):
            c = "https:" + c
        elif c.startswith("/") and base_url:
            c = urllib.parse.urljoin(base_url, c)
        elif not c.startswith("http") and base_url:
            c = urllib.parse.urljoin(base_url, c)
        resolved.append(c)

    return list(set(resolved))


def score_candidate(candidate_url: str, original_url: str) -> int:
    """Score a candidate URL — higher is better."""
    score = 0
    lower = candidate_url.lower()

    good_kw = [
        "cam", "webcam", "snapshot", "live", "current", "latest", "stream",
        "frame", "image", "photo", "still", "capture",
    ]
    if any(kw in lower for kw in good_kw):
        score += 3

    bad_kw = [
        "logo", "icon", "thumb", "avatar", "favicon", "banner", "button",
        "arrow", "spacer", "pixel", "tracking", "ad", "1x1",
    ]
    if any(kw in lower for kw in bad_kw):
        score -= 5

    ad_domains = ["doubleclick", "googleads", "googlesyndication", "facebook.com/tr"]
    if any(ad in lower for ad in ad_domains):
        score -= 5

    if original_url and str(original_url) != "nan":
        orig_domain = get_domain(original_url)
        if orig_domain and orig_domain in candidate_url:
            score += 2
        orig_file = original_url.split("?")[0].split("/")[-1]
        cand_file = candidate_url.split("?")[0].split("/")[-1]
        if orig_file and orig_file == cand_file:
            score += 1

    return score


# ---------------------------------------------------------------------------
# Agentic repair strategies
# ---------------------------------------------------------------------------

def _try_direct_revalidation(url: str) -> tuple:
    """Strategy 0: Re-check the original URL (Agent 1 might have had a transient failure)."""
    valid, reason, _ = validate_image_url(url, timeout=(15, 20))
    if valid:
        return True, url, "direct_revalidation"
    return False, None, None


def _try_strip_query_params(url: str) -> tuple:
    """Strategy 1: Strip query string (cache-busting timestamps, etc.)."""
    if "?" not in url:
        return False, None, None
    base = url.split("?")[0]
    valid, _, _ = validate_image_url(base)
    if valid:
        return True, base, "strip_query"
    return False, None, None


def _try_scrape_source(url_source: str, original_url: str) -> tuple:
    """Strategy 2: Scrape the source page for working image URLs."""
    if not url_source or not url_source.startswith("http"):
        return False, None, None
    url_source = normalize_url(url_source)
    domain = get_domain(url_source)
    if not domain:
        return False, None, None

    rate_limiter.wait(domain)
    try:
        resp = requests.get(
            url_source, timeout=(10, 15),
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebcamAgent/1.0)"},
            verify=False,
        )
        if resp.status_code != 200:
            return False, None, None

        candidates = find_urls_in_html(resp.text, base_url=url_source)
        scored = [(score_candidate(c, original_url), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        for _score, c in scored[:8]:
            if _score < -2:
                continue
            valid, _, _ = validate_image_url(c)
            if valid:
                return True, c, "scrape"
    except Exception as e:
        logger.debug(f"Scrape failed for {url_source}: {e}")

    return False, None, None


def _try_wayback(url: str) -> tuple:
    """Strategy 3: Check the Wayback Machine for an archived snapshot."""
    try:
        wb_api = f"https://archive.org/wayback/available?url={urllib.parse.quote(url)}"
        resp = requests.get(wb_api, timeout=(10, 15))
        if resp.status_code != 200:
            return False, None, None
        data = resp.json()
        snap = data.get("archived_snapshots", {}).get("closest", {})
        if snap.get("available"):
            snap_url = snap["url"]
            valid, _, _ = validate_image_url(snap_url)
            if valid:
                return True, snap_url, "wayback"
    except Exception as e:
        logger.debug(f"Wayback error: {e}")
    return False, None, None


def _try_pattern_mutations(url: str) -> tuple:
    """Strategy 4: Try common URL pattern mutations."""
    mutations = []

    # Timestamp refresh
    if "?" in url:
        mutations.append(url.split("?")[0] + f"?t={int(time.time())}")

    base = url.rsplit("/", 1)[0]
    for name in [
        "latest.jpg", "current.jpg", "snapshot.jpg", "live.jpg", "webcam.jpg",
        "image.jpg", "latest-frame.jpg", "frame.jpg", "still.jpg",
    ]:
        mutations.append(f"{base}/{name}")

    for m in mutations:
        valid, _, _ = validate_image_url(m)
        if valid:
            return True, m, "pattern_mutation"
    return False, None, None


def _try_faa_repair(url: str) -> tuple:
    """Strategy 5: FAA-specific API lookup for weathercam URLs."""
    if "wcams-static.faa.gov" not in url and "weathercams.faa.gov" not in url:
        return False, None, None

    # Try API endpoint
    match = re.search(r"webimages/([^/]+)/([^/]+)/([^/]+)-\d+\.jpg", url)
    if match:
        site_id, cam_id, image_id = match.groups()
        api_url = f"https://weathercams.faa.gov/map/cameraSite/{site_id}/details/camera/{image_id}/full"
        try:
            resp = requests.get(api_url, timeout=(10, 15), verify=False)
            if resp.status_code == 200:
                data = resp.json()
                latest = data.get("url") or data.get("latestImageUrl")
                if latest:
                    valid, _, _ = validate_image_url(latest)
                    if valid:
                        return True, latest, "faa_api"
        except Exception:
            pass

    # Try site listing
    match = re.search(r"webimages/([^/]+)/", url)
    if match:
        site_id = match.group(1)
        site_url = f"https://weathercams.faa.gov/map/cameraSite/{site_id}/details"
        domain = get_domain(site_url)
        if domain:
            rate_limiter.wait(domain)
            try:
                resp = requests.get(site_url, timeout=(10, 15), verify=False)
                if resp.status_code == 200:
                    candidates = find_urls_in_html(resp.text, base_url=site_url)
                    for c in candidates:
                        valid, _, _ = validate_image_url(c)
                        if valid:
                            return True, c, "faa_scrape"
            except Exception:
                pass

    return False, None, None


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_row(args):
    """Agentic multi-strategy repair for a single row."""
    index, row = args

    if pd.notna(row.get("repair_status")):
        return None  # Already processed

    orig_url = str(row.get("ImageURL", ""))
    h_status = str(row.get("health_status", ""))

    if h_status == "VALID":
        return (index, "ALREADY_VALID", "", 0, orig_url)

    url_source = str(row.get("URLSource", ""))
    h_reason = str(row.get("health_failure_reason", ""))
    attempts = 0

    # --- Agentic strategy selection based on failure reason ---
    # Build an ordered list of strategies to try
    strategies = []

    if h_reason in ("timeout_connect", "timeout_read", "connection_error", "unknown_error"):
        # Transient failure — re-check first
        strategies.append(_try_direct_revalidation)

    if "?" in orig_url:
        strategies.append(lambda u: _try_strip_query_params(u))

    # FAA-specific before generic strategies
    if "faa.gov" in orig_url:
        strategies.append(lambda u: _try_faa_repair(u))

    # Source scraping
    if url_source and url_source.lower() != "nan" and url_source != "faa.gov":
        strategies.append(lambda u: _try_scrape_source(url_source, u))

    # Wayback Machine (skip for empty/dns-failure — URL itself is likely gone)
    if h_reason not in ("empty", "dns_failure"):
        strategies.append(lambda u: _try_wayback(u))

    # Pattern mutations
    strategies.append(lambda u: _try_pattern_mutations(u))

    for strategy_fn in strategies:
        attempts += 1
        try:
            found, repaired_url, method = strategy_fn(orig_url)
            if found:
                return (index, "REPAIRED", method, attempts, repaired_url)
        except Exception as e:
            logger.debug(f"Strategy error row {index}: {e}")

    return (index, "UNRESOLVED", "", attempts, orig_url)


def main(csv_path: str, sample: int = None):
    """Main execution of Agent 2."""
    global g_df, g_csv_path, g_interrupted
    g_interrupted = False
    signal.signal(signal.SIGINT, sigint_handler)

    g_csv_path = csv_path

    try:
        df = pd.read_csv(csv_path)
        for col in ["original_image_url", "repair_status", "repair_method",
                     "repair_attempts", "repair_timestamp"]:
            if col not in df.columns:
                df[col] = pd.NA
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    if sample and len(df) > sample:
        df = df.head(sample)

    g_df = df
    indices_to_check = df[df["repair_status"].isna()].index.tolist()

    if not indices_to_check:
        print("[Agent 2] All rows already processed. Skipping.")
        safe_csv_write(g_df, g_csv_path)
        return

    logger.info(f"Agent 2: Repairing {len(indices_to_check)} URLs.")

    processed = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        tasks = [(idx, g_df.loc[idx]) for idx in indices_to_check]
        future_to_idx = {executor.submit(process_row, t): t[0] for t in tasks}

        for future in tqdm(as_completed(future_to_idx), total=len(indices_to_check),
                           desc="Agent 2"):
            if g_interrupted:
                break

            idx = future_to_idx[future]
            try:
                res = future.result()
                if res is not None:
                    _, r_status, r_method, r_attempts, r_url = res
                    g_df.at[idx, "repair_status"] = r_status
                    g_df.at[idx, "repair_method"] = r_method
                    g_df.at[idx, "repair_attempts"] = r_attempts
                    g_df.at[idx, "repair_timestamp"] = datetime.utcnow().isoformat() + "Z"
                    if r_status == "REPAIRED":
                        # Preserve the original URL before overwriting
                        g_df.at[idx, "original_image_url"] = g_df.at[idx, "ImageURL"]
                        g_df.at[idx, "ImageURL"] = r_url
            except Exception as e:
                logger.error("Error repairing row", extra={"row": idx, "detail": str(e)})

            processed += 1
            if processed % 100 == 0:
                safe_csv_write(g_df, g_csv_path)

    safe_csv_write(g_df, g_csv_path)

    # Print summary
    total = len(g_df)
    repaired = len(g_df[g_df["repair_status"] == "REPAIRED"])
    already_valid = len(g_df[g_df["repair_status"] == "ALREADY_VALID"])
    unresolved = len(g_df[g_df["repair_status"] == "UNRESOLVED"])

    print("\n" + " Agent 2 Summary ".center(40, "="))
    print(f"{'Status'.ljust(15)} | {'Count'.rjust(8)} | {'%'.rjust(8)}")
    print("-" * 40)
    print(f"{'REPAIRED'.ljust(15)} | {str(repaired).rjust(8)} | {f'{(repaired/total*100):.1f}'.rjust(8)}")
    print(f"{'ALREADY_VALID'.ljust(15)} | {str(already_valid).rjust(8)} | {f'{(already_valid/total*100):.1f}'.rjust(8)}")
    print(f"{'UNRESOLVED'.ljust(15)} | {str(unresolved).rjust(8)} | {f'{(unresolved/total*100):.1f}'.rjust(8)}")
    print(f"{'TOTAL'.ljust(15)} | {str(total).rjust(8)} | {'100.0'.rjust(8)}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="webcam_list.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(args.csv, args.sample)
