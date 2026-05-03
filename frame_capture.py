"""frame_capture.py - Periodic webcam frame capture
====================================================

Downloads one frame at a fixed cadence (1 frame / minute by default) from
a single webcam URL for a fixed duration (24 hours by default), writing
each frame to disk with a UTC timestamp.

Folder layout
-------------
The output is laid out so that one camera, one session, one calendar day
each get their own clean container:

    <output_root>/                          # default: ./captures
        <camera_slug>/                      # derived from --name or URL host
            <session_start_iso>/            # 2026-05-03T20-30-00Z
                frames/
                    <YYYY-MM-DD>/
                        HH-MM-SSZ_NNNNN.<ext>
                    <YYYY-MM-DD>/           # next-day folder when crossing midnight
                        HH-MM-SSZ_NNNNN.<ext>
                manifest.jsonl              # one JSON line per attempted frame
                session.json                # config snapshot + final summary
                capture.log                 # human-readable log

Properties
----------
* Drift-free scheduling. Tick `n` is anchored at session_start + n*interval
  using `time.monotonic()`, so a slow individual download never causes the
  whole capture to fall behind.
* Skip on overrun. If a download takes longer than `interval`, the missed
  ticks are recorded as `skipped_overrun` and capture jumps to the next
  future tick (no pile-up).
* Per-frame retries with linear backoff and a connection / read timeout.
* Real image verification. A frame is only counted as `ok` if the bytes
  decode through PIL (`Image.open(...).verify()`).
* Atomic writes. Each frame is written to `*.tmp` and `os.replace`-d into
  place, so an interrupted run never leaves a half-written image.
* Streaming manifest. Each manifest line is flushed and fsync-d so the
  exact set of saved frames is recoverable even after a hard crash.
* Clean Ctrl+C. SIGINT finishes the current tick, flushes manifest +
  summary, and exits with status 0.

CLI
---
    conda activate cg
    python frame_capture.py --url https://example.com/cam.jpg
    python frame_capture.py --url https://example.com/cam.jpg --duration 5m --interval 10s
    python frame_capture.py --url https://example.com/cam.jpg --duration 24h --name myroof
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import errno
import signal
import hashlib
import logging
import argparse
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlsplit

import requests
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; FrameCapture/1.0)"
DEFAULT_OUTPUT_ROOT = "captures"

# Magic-byte -> file extension. Order matters (longer signatures first).
EXT_FROM_MAGIC: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xFF\xD8\xFF",      ".jpg"),
    (b"GIF87a",            ".gif"),
    (b"GIF89a",            ".gif"),
    (b"BM",                ".bmp"),
    (b"II*\x00",           ".tif"),
    (b"MM\x00*",           ".tif"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_iso(dt: datetime) -> str:
    """Filesystem-safe UTC ISO-8601 (no colons)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _utc_iso_strict(dt: datetime) -> str:
    """Strict ISO-8601 with colons (for inside JSON / logs)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(s: str, max_len: int = 60) -> str:
    """Make a filesystem-safe, lower-case slug from a URL or arbitrary name."""
    s = (s or "").strip()
    # Strip scheme; keep host + first path segment
    if "://" in s:
        parts = urlsplit(s)
        host = parts.netloc.split(":")[0]
        first_path = next((p for p in parts.path.split("/") if p), "")
        s = f"{host}_{first_path}" if first_path else host
    s = s.lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s).strip("-_.")
    s = re.sub(r"-{2,}", "-", s)
    return (s[:max_len] or "camera").rstrip("-_.")


def _detect_extension(data: bytes) -> str:
    for sig, ext in EXT_FROM_MAGIC:
        if data.startswith(sig):
            return ext
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _human_bytes(n: int) -> str:
    if n is None:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    f = float(n); i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0; i += 1
    return f"{f:,.1f} {units[i]}" if i > 0 else f"{int(f)} {units[i]}"


def _human_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s2 = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s2:02d}s" if h else (f"{m:d}m{s2:02d}s" if m else f"{s:d}s")


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])?\s*$", re.IGNORECASE)
_UNIT_FACTOR = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, None: 1.0}


def parse_duration(s: str) -> float:
    """Parse '24h', '5m', '30s', '0.5h', or a bare number-of-seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    m = _DURATION_RE.match(str(s))
    if not m:
        raise argparse.ArgumentTypeError(
            f"Cannot parse duration {s!r}. Use e.g. 24h, 5m, 30s, or seconds.")
    val = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    return val * _UNIT_FACTOR[unit]


# ---------------------------------------------------------------------------
# Capture session
# ---------------------------------------------------------------------------
class CaptureSession:
    """One on-disk capture session (a single output directory).

    The session_dir is created up-front; manifest.jsonl and session.json are
    appended/written incrementally so the run is recoverable mid-session.
    """

    def __init__(
        self,
        url: str,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        name: Optional[str] = None,
        interval_s: float = 60.0,
        duration_s: float = 86400.0,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 15.0,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        if not url or "://" not in url:
            raise ValueError(f"URL must include a scheme (http/https): {url!r}")
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.url = url.strip()
        self.slug = _slugify(name or self.url)
        self.interval_s = float(interval_s)
        self.duration_s = float(duration_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.read_timeout_s = float(read_timeout_s)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.user_agent = user_agent

        self.start_dt = _utc_now()
        self.session_dir = (Path(output_root) / self.slug / _utc_iso(self.start_dt)).resolve()
        self.frames_dir = self.session_dir / "frames"
        self.manifest_path = self.session_dir / "manifest.jsonl"
        self.session_path = self.session_dir / "session.json"
        self.log_path = self.session_dir / "capture.log"

        # Counters tracked across the run for summary.
        self.n_attempted = 0
        self.n_ok = 0
        self.n_failed = 0
        self.n_skipped_overrun = 0
        self.bytes_total = 0
        self.unique_hashes: set = set()
        self.n_duplicates = 0
        self.last_hash: Optional[str] = None

        self._stop = threading.Event()
        self._installed_sigint = False
        self._previous_sigint = None

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self._write_session_json(final=False)

    # --- logging ---------------------------------------------------------
    def _setup_logger(self):
        self.logger = logging.getLogger(f"frame_capture.{self.slug}")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers when re-instantiating for tests
        self.logger.handlers = []
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                 datefmt="%Y-%m-%dT%H:%M:%SZ")
        # Timestamps in UTC
        logging.Formatter.converter = time.gmtime
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.propagate = False

    # --- signal handling -------------------------------------------------
    def _install_sigint(self):
        def handler(signum, frame):
            if self._stop.is_set():
                # Second Ctrl+C: hard exit
                self.logger.warning("Second SIGINT; exiting immediately.")
                sys.exit(130)
            self._stop.set()
            self.logger.warning("SIGINT received; finishing current tick and "
                                "writing summary. Press Ctrl+C again to abort.")
        try:
            self._previous_sigint = signal.signal(signal.SIGINT, handler)
            self._installed_sigint = True
        except ValueError:
            # Not main thread; ignore
            pass

    def _restore_sigint(self):
        if self._installed_sigint and self._previous_sigint is not None:
            try: signal.signal(signal.SIGINT, self._previous_sigint)
            except Exception: pass

    # --- networking ------------------------------------------------------
    def _attempt_download(self) -> Tuple[Optional[bytes], Optional[int], Optional[str]]:
        """Single HTTP GET attempt. Returns (bytes_or_None, http_code, error_or_None)."""
        try:
            resp = requests.get(
                self.url,
                timeout=(self.connect_timeout_s, self.read_timeout_s),
                headers={"User-Agent": self.user_agent},
                verify=False,
            )
        except requests.exceptions.ConnectTimeout:
            return None, None, "timeout_connect"
        except requests.exceptions.ReadTimeout:
            return None, None, "timeout_read"
        except requests.exceptions.SSLError:
            return None, None, "ssl_error"
        except requests.exceptions.TooManyRedirects:
            return None, None, "too_many_redirects"
        except requests.exceptions.ConnectionError as e:
            msg = str(e).lower()
            if "name or service" in msg or "getaddrinfo" in msg:
                return None, None, "dns_failure"
            return None, None, "connection_error"
        except Exception as e:
            return None, None, f"unknown_error:{type(e).__name__}"

        code = resp.status_code
        if code != 200:
            return None, code, f"http_{code}"
        if not resp.content:
            return None, code, "empty_response"
        # Verify it's a real image so we don't save HTML "we're offline" pages
        try:
            with Image.open(BytesIO(resp.content)) as img:
                img.verify()
        except Exception:
            return None, code, "not_an_image"
        return resp.content, code, None

    def _capture_with_retries(self) -> Tuple[Optional[bytes], int, Optional[int], Optional[str]]:
        """Returns (bytes_or_None, attempts, last_http_code, last_error)."""
        last_code, last_err = None, None
        for attempt in range(1, self.max_retries + 1):
            data, code, err = self._attempt_download()
            if data is not None:
                return data, attempt, code, None
            last_code, last_err = code, err
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_s * attempt)
        return None, self.max_retries, last_code, last_err

    # --- per-frame disk operations --------------------------------------
    def _save_frame_atomic(self, data: bytes, scheduled_dt: datetime, seq: int
                           ) -> Tuple[Path, str]:
        """Write the frame to <frames_dir>/<date>/HH-MM-SSZ_NNNNN.ext atomically.

        Returns (final_path, posix_relative_path).
        """
        date_dir = self.frames_dir / scheduled_dt.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        ext = _detect_extension(data)
        fname = f"{scheduled_dt.strftime('%H-%M-%SZ')}_{seq:05d}{ext}"
        final_path = date_dir / fname
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        rel = final_path.relative_to(self.session_dir).as_posix()
        return final_path, rel

    def _append_manifest(self, entry: Dict[str, Any]):
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        # Use append+fsync so manifest survives a hard crash
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try: os.fsync(f.fileno())
            except OSError: pass

    # --- session metadata ------------------------------------------------
    def _write_session_json(self, final: bool):
        payload = {
            "schema": "frame_capture/1",
            "url": self.url,
            "camera_slug": self.slug,
            "session_dir": str(self.session_dir),
            "session_start_utc": _utc_iso_strict(self.start_dt),
            "interval_s": self.interval_s,
            "duration_s": self.duration_s,
            "expected_frames": int(round(self.duration_s / self.interval_s)),
            "user_agent": self.user_agent,
            "connect_timeout_s": self.connect_timeout_s,
            "read_timeout_s": self.read_timeout_s,
            "max_retries": self.max_retries,
            "retry_backoff_s": self.retry_backoff_s,
            "complete": final,
        }
        if final:
            payload.update({
                "session_end_utc": _utc_iso_strict(_utc_now()),
                "n_attempted": self.n_attempted,
                "n_ok": self.n_ok,
                "n_failed": self.n_failed,
                "n_skipped_overrun": self.n_skipped_overrun,
                "n_unique_hashes": len(self.unique_hashes),
                "n_duplicates": self.n_duplicates,
                "bytes_total": self.bytes_total,
                "bytes_total_human": _human_bytes(self.bytes_total),
            })
        tmp = self.session_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.session_path)

    # --- main loop -------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Capture frames until the duration is reached or SIGINT arrives."""
        self._install_sigint()
        try:
            return self._run_inner()
        finally:
            self._write_session_json(final=True)
            self._restore_sigint()

    def _run_inner(self) -> Dict[str, Any]:
        total_ticks = max(1, int(round(self.duration_s / self.interval_s)))
        self.logger.info(
            "Starting capture: url=%s slug=%s interval=%.1fs duration=%s ticks=%d",
            self.url, self.slug, self.interval_s,
            _human_duration(self.duration_s), total_ticks,
        )
        self.logger.info("Session dir: %s", self.session_dir)

        t0 = time.monotonic()
        n = 0
        while n < total_ticks and not self._stop.is_set():
            target_t = t0 + n * self.interval_s
            scheduled_dt = self.start_dt + timedelta(seconds=n * self.interval_s)
            now_t = time.monotonic()

            # --- wait or detect overrun
            sleep_for = target_t - now_t
            if sleep_for > 0:
                # Wake periodically so SIGINT is responsive
                self._stop.wait(timeout=min(sleep_for, 1.0))
                while not self._stop.is_set() and time.monotonic() < target_t:
                    self._stop.wait(timeout=min(target_t - time.monotonic(), 1.0))
                if self._stop.is_set():
                    break
            else:
                lateness = -sleep_for
                if lateness >= self.interval_s:
                    n_missed = int(lateness // self.interval_s)
                    for k in range(n_missed):
                        miss_seq = n + k
                        miss_dt = self.start_dt + timedelta(seconds=miss_seq * self.interval_s)
                        miss_entry = {
                            "seq": miss_seq,
                            "scheduled_utc": _utc_iso_strict(miss_dt),
                            "status": "skipped_overrun",
                        }
                        self.n_skipped_overrun += 1
                        self._append_manifest(miss_entry)
                        self.logger.warning(
                            "tick %d skipped (overrun by %.1fs)", miss_seq, lateness)
                    n += n_missed
                    continue

            # --- capture
            self.n_attempted += 1
            t_start = time.monotonic()
            data, attempts, http_code, err = self._capture_with_retries()
            duration_ms = int(round((time.monotonic() - t_start) * 1000))
            completed_dt = _utc_now()

            entry: Dict[str, Any] = {
                "seq": n,
                "scheduled_utc": _utc_iso_strict(scheduled_dt),
                "completed_utc": _utc_iso_strict(completed_dt),
                "attempts": attempts,
                "http_code": http_code,
                "duration_ms": duration_ms,
            }

            if data is None:
                entry["status"] = "failed"
                entry["error"] = err
                self.n_failed += 1
                self._append_manifest(entry)
                self.logger.error(
                    "tick %d/%d FAILED after %d attempt(s): %s (http=%s, %dms)",
                    n + 1, total_ticks, attempts, err, http_code, duration_ms)
            else:
                # Hash + image probe
                sha = hashlib.sha256(data).hexdigest()
                size = len(data)
                width = height = None
                fmt = None
                try:
                    with Image.open(BytesIO(data)) as img:
                        fmt = (img.format or "").lower()
                        width, height = img.size
                except Exception:
                    pass
                final_path, rel_path = self._save_frame_atomic(data, scheduled_dt, n)
                duplicate = (sha == self.last_hash)
                if duplicate:
                    self.n_duplicates += 1
                self.last_hash = sha
                self.unique_hashes.add(sha)
                self.bytes_total += size
                self.n_ok += 1
                entry.update({
                    "status": "ok",
                    "path": rel_path,
                    "size_bytes": size,
                    "sha256": sha,
                    "image_format": fmt,
                    "width": width,
                    "height": height,
                    "duplicate_of_previous": duplicate,
                })
                self._append_manifest(entry)
                self.logger.info(
                    "tick %d/%d ok %s %s %sx%s in %dms (attempt %d) -> %s%s",
                    n + 1, total_ticks, _human_bytes(size),
                    fmt or "?", width, height, duration_ms, attempts, rel_path,
                    "  [duplicate]" if duplicate else "",
                )

            # --- periodic session.json refresh (every 20 ticks)
            if (n + 1) % 20 == 0:
                self._write_session_json(final=False)

            n += 1

        # --- post-loop summary
        self._print_summary(stopped_early=self._stop.is_set())
        return self._summary_dict()

    def _summary_dict(self) -> Dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "n_attempted": self.n_attempted,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_skipped_overrun": self.n_skipped_overrun,
            "n_unique_hashes": len(self.unique_hashes),
            "n_duplicates": self.n_duplicates,
            "bytes_total": self.bytes_total,
        }

    def _print_summary(self, stopped_early: bool):
        total_ticks = max(1, int(round(self.duration_s / self.interval_s)))
        ok_pct = 100.0 * self.n_ok / total_ticks if total_ticks else 0.0
        bar = "=" * 64
        lines = [
            "",
            bar,
            (" Capture finished " if not stopped_early else " Capture stopped (SIGINT) ").center(64, "="),
            bar,
            f"  Camera URL    : {self.url}",
            f"  Camera slug   : {self.slug}",
            f"  Session dir   : {self.session_dir}",
            f"  Started (UTC) : {_utc_iso_strict(self.start_dt)}",
            f"  Ended   (UTC) : {_utc_iso_strict(_utc_now())}",
            f"  Interval      : {self.interval_s:.1f}s",
            f"  Target frames : {total_ticks}",
            f"  Saved (ok)    : {self.n_ok}  ({ok_pct:.1f}% of target)",
            f"  Failed        : {self.n_failed}",
            f"  Skipped       : {self.n_skipped_overrun}",
            f"  Total bytes   : {_human_bytes(self.bytes_total)}",
            f"  Unique frames : {len(self.unique_hashes)}",
            f"  Duplicates    : {self.n_duplicates}",
            bar,
        ]
        for line in lines:
            self.logger.info(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download one frame at a fixed cadence (default: 1 frame "
                    "per minute for 24 hours) from a webcam URL.",
        epilog=(
            "Examples:\n"
            "  conda activate cg\n"
            "  python frame_capture.py --url https://example.com/cam.jpg\n"
            "  python frame_capture.py --url https://example.com/cam.jpg --duration 5m --interval 10s\n"
            "  python frame_capture.py --url https://example.com/cam.jpg --duration 24h --name myroof\n"
        ),
    )
    p.add_argument("--url", required=True, help="Camera image URL (http/https).")
    p.add_argument("--duration", type=parse_duration, default="24h",
                   help="Total capture duration (e.g. 24h, 5m, 30s). Default: 24h.")
    p.add_argument("--interval", type=parse_duration, default="60s",
                   help="Time between consecutive frames (e.g. 60s, 1m). Default: 60s.")
    p.add_argument("--out", default=DEFAULT_OUTPUT_ROOT,
                   help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}/")
    p.add_argument("--name", default=None,
                   help="Camera slug. Default: derived from URL host.")
    p.add_argument("--connect-timeout", type=float, default=10.0,
                   help="HTTP connect timeout in seconds. Default: 10.")
    p.add_argument("--read-timeout", type=float, default=15.0,
                   help="HTTP read timeout in seconds. Default: 15.")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Number of attempts per frame (>= 1). Default: 2.")
    p.add_argument("--retry-backoff", type=float, default=2.0,
                   help="Linear backoff between retries (s). Default: 2.0.")
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                   help="User-Agent header to send.")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    sess = CaptureSession(
        url=args.url, output_root=args.out, name=args.name,
        interval_s=args.interval, duration_s=args.duration,
        connect_timeout_s=args.connect_timeout,
        read_timeout_s=args.read_timeout,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff,
        user_agent=args.user_agent,
    )
    sess.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
