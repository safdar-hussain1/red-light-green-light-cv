"""Download the YOLO11n person-detection weights into models/.

Tries the pinned Ultralytics release asset first; if that URL is gone,
falls back to letting the ultralytics package resolve the download, then
moves the file into place. Prints size and sha256 so the artifact is
auditable.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
DEST = Path(__file__).resolve().parent.parent / "models" / "yolo11n.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DEST.exists():
        print(f"already present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB, sha256 {sha256(DEST)})")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"downloading {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, DEST)
    except urllib.error.URLError as exc:
        print(f"pinned URL failed ({exc}); falling back to ultralytics auto-download")
        from ultralytics import YOLO

        YOLO("yolo11n.pt")  # downloads into the current directory
        shutil.move("yolo11n.pt", DEST)
    size_mb = DEST.stat().st_size / 1e6
    if not 4.0 <= size_mb <= 8.0:
        print(f"error: unexpected weights size {size_mb:.1f} MB", file=sys.stderr)
        return 1
    print(f"saved {DEST} ({size_mb:.1f} MB, sha256 {sha256(DEST)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
