import os
import json
import logging
import threading
import time
import urllib.parse
from datetime import datetime
from io import BytesIO

import urllib3
import requests
from PIL import Image, ImageOps, ImageStat

urllib3.disable_warnings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure plain and structured logging."""
    logger = logging.getLogger("webcam_agent")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    plain_handler = logging.FileHandler("webcam_agent.log", encoding="utf-8")
    plain_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    plain_handler.setFormatter(plain_formatter)
    logger.addHandler(plain_handler)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "agent": record.module,
                "row": getattr(record, "row", -1),
                "event": record.msg,
                "detail": getattr(record, "detail", "")
            }
            return json.dumps(log_obj)

    json_handler = logging.FileHandler("webcam_agent_structured.jsonl", encoding="utf-8")
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)

    return logger

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-domain rate limiter."""
    def __init__(self):
        self._last_requested = {}
        self._lock = threading.Lock()

    def wait(self, domain: str, min_interval_seconds: float = 2.0):
        with self._lock:
            now = time.time()
            last = self._last_requested.get(domain, 0.0)
            elapsed = now - last
            if elapsed < min_interval_seconds:
                time.sleep(min_interval_seconds - elapsed)
            self._last_requested[domain] = time.time()

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Normalize a given URL string."""
    if not isinstance(url, str):
        return ""
    url = url.replace("\xA0", " ").strip()
    url = url.strip("\"'")
    if url.startswith("//"):
        url = "https:" + url
    return url


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urllib.parse.urlparse(normalize_url(url)).netloc
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

_IMAGE_SIGNATURES = [
    (b"\xFF\xD8\xFF",              "jpeg"),
    (b"\x89PNG\x0D\x0A\x1A\x0A",  "png"),
    (b"GIF87a",                    "gif"),
    (b"GIF89a",                    "gif"),
    (b"BM",                        "bmp"),
    (b"\x49\x49\x2A\x00",         "tiff_le"),
    (b"\x4D\x4D\x00\x2A",         "tiff_be"),
]


def _check_magic_bytes(data: bytes) -> bool:
    """Return True if *data* starts with known image magic bytes."""
    for sig, _ in _IMAGE_SIGNATURES:
        if data.startswith(sig):
            return True
    # WEBP: starts with RIFF, bytes 8-12 are "WEBP"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def validate_image_url(url: str, timeout=(10, 15)) -> tuple:
    """Validate whether *url* points to a reachable image.

    Returns ``(is_valid, reason, http_status_code)``.
    *reason* is ``"ok"`` when valid; otherwise a short diagnostic string.
    *http_status_code* is 0 when the request never reached the server.
    """
    if not url or str(url).lower() == "nan" or str(url).strip() == "":
        return False, "empty", 0

    url = normalize_url(url)
    if not url.startswith(("http://", "https://")):
        return False, "invalid_scheme", 0

    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebcamAgent/1.0)"},
            verify=False,
        )
        status = resp.status_code
        if status != 200:
            return False, f"http_{status}", status

        # Read first 32 bytes — enough for all known magic byte checks
        chunk = resp.raw.read(32)
        if not chunk or len(chunk) < 2:
            return False, "empty_response", status

        if _check_magic_bytes(chunk):
            return True, "ok", status

        return False, "not_an_image", status

    except requests.exceptions.TooManyRedirects:
        return False, "redirect_limit", 0
    except requests.exceptions.ConnectionError as e:
        err_str = str(e).lower()
        if "name or service not known" in err_str or "getaddrinfo failed" in err_str:
            return False, "dns_failure", 0
        return False, "connection_error", 0
    except requests.exceptions.ConnectTimeout:
        return False, "timeout_connect", 0
    except requests.exceptions.ReadTimeout:
        return False, "timeout_read", 0
    except Exception:
        return False, "unknown_error", 0

# ---------------------------------------------------------------------------
# Image download & quality
# ---------------------------------------------------------------------------

_DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WebcamAgent/1.0)"}


def download_image(url: str, timeout=(10, 15)) -> Image.Image:
    """Download image and fix EXIF orientation.  Returns None on failure."""
    try:
        url = normalize_url(url)
        resp = requests.get(url, timeout=timeout, headers=_DOWNLOAD_HEADERS, verify=False)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        return None


def download_image_with_retry(url: str, max_retries: int = 3) -> Image.Image:
    """Agentic download: retry with increasing timeouts on failure."""
    for attempt in range(1, max_retries + 1):
        timeout = (10 * attempt, 15 * attempt)
        img = download_image(url, timeout=timeout)
        if img is not None:
            return img
    return None


def assess_image_quality(img: Image.Image) -> tuple:
    """Assess whether an image is suitable for calibration.

    Returns ``(is_suitable, reason)``.
    """
    if img is None:
        return False, "no_image"

    w, h = img.size
    if w < 100 or h < 100:
        return False, "too_small"

    stat = ImageStat.Stat(img)
    avg_std = sum(stat.stddev) / len(stat.stddev)
    if avg_std < 3.0:
        return False, "solid_color_or_blank"

    avg_mean = sum(stat.mean) / len(stat.mean)
    if avg_mean < 8:
        return False, "too_dark"
    if avg_mean > 250:
        return False, "overexposed"

    return True, "ok"


def is_image_too_small(img: Image.Image, min_px: int = 50) -> bool:
    """Check if image dimensions are below minimum threshold."""
    if img is None:
        return True
    return img.width < min_px or img.height < min_px

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def safe_csv_write(df, path: str):
    """Write DataFrame atomically via a temporary file."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
