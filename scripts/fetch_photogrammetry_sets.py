"""
AXUM ROVER — Photogrammetry Test Set Fetcher
=============================================
Downloads known-good open image sets into ``scans/photos/TEST-*/`` so the
Meshroom pipeline can be exercised and tuned without waiting on a real
turntable capture.

TEST-SCEAUX (openMVG's Sceaux Castle) is already committed and is what the
pipeline was first validated against. The sets here add harder cases:
Monstree is a small textured object shot handheld — much closer to an
artefact capture than a building facade — and Buddha is a dense
turntable-style set of a sculpted object, which is the closest public
analogue to what the rover actually does.

Why GitHub repositories rather than a dataset host: these are the sets
AliceVision and openMVG themselves ship as pipeline references, so a
failure on them is a pipeline bug rather than a bad-input problem. Each
repo is downloaded as a codeload zip — cloning would pull the full history
for what is a one-shot fetch of image files.

Sizes are real and worth reading before starting: Buddha is ~780 MB.

Run from project root (venv activated):
  python scripts/fetch_photogrammetry_sets.py --list
  python scripts/fetch_photogrammetry_sets.py --set monstree
  python scripts/fetch_photogrammetry_sets.py --all

Writes:
  scans/photos/TEST-<NAME>/*.jpg
  scans/photos/TEST-<NAME>/SOURCE.json
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SCAN_PHOTOS_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REQUEST_TIMEOUT = 120


@dataclass(frozen=True)
class PhotoSet:
    """One downloadable reference image set."""

    key: str
    folder: str
    repo: str
    branch: str
    licence: str
    approx_mb: int
    note: str

    @property
    def zip_url(self) -> str:
        return f"https://codeload.github.com/{self.repo}/zip/refs/heads/{self.branch}"

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.repo}"


# Verified against the GitHub API before being listed here — a fetcher that
# 404s halfway through is worse than one that offers fewer sets.
PHOTO_SETS: tuple[PhotoSet, ...] = (
    PhotoSet(
        key="sceaux",
        folder="TEST-SCEAUX",
        repo="openMVG/ImageDataset_SceauxCastle",
        branch="master",
        licence="See repository",
        approx_mb=12,
        note="Building facade. Already committed as the pipeline's first test fixture.",
    ),
    PhotoSet(
        key="monstree",
        folder="TEST-MONSTREE",
        repo="alicevision/dataset_monstree",
        branch="master",
        licence="See repository",
        approx_mb=164,
        note="Small textured object, handheld. Closest to an artefact capture.",
    ),
    PhotoSet(
        key="buddha",
        folder="TEST-BUDDHA",
        repo="alicevision/dataset_buddha",
        branch="master",
        licence="CC-BY-4.0",
        approx_mb=780,
        note="Dense sculpted-object set. Closest public analogue to turntable capture.",
    ),
)

SETS_BY_KEY = {photo_set.key: photo_set for photo_set in PHOTO_SETS}


def _extract_images(archive: zipfile.ZipFile, target: Path) -> int:
    """
    Flatten every image in a repo zip into one folder.

    Meshroom is pointed at a single directory of photos, but these repos
    nest images under a branch-named root and sometimes a subfolder. The
    flattening is why names are prefixed with their parent directory --
    two subfolders each containing ``001.jpg`` would otherwise collide and
    silently halve the input set.

    Args:
        archive: Opened repository zip
        target:  Destination photo directory

    Returns:
        Number of images written
    """
    written = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        source = Path(info.filename)
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = source.parts[1:]  # drop the "<repo>-<branch>/" root
        if not parts:
            continue
        prefix = "_".join(parts[:-1])
        name = f"{prefix}_{source.name}" if prefix else source.name

        with archive.open(info) as handle, open(target / name, "wb") as out:
            shutil.copyfileobj(handle, out)
        written += 1
    return written


def fetch_set(photo_set: PhotoSet, overwrite: bool) -> bool:
    """
    Download and unpack one image set.

    Args:
        photo_set: Set definition
        overwrite: Replace an existing folder

    Returns:
        True when images were written
    """
    target = SCAN_PHOTOS_DIR / photo_set.folder
    existing = list(target.glob("*")) if target.exists() else []
    if existing and not overwrite:
        logger.info(f"Skip (exists, {len(existing)} files): {photo_set.folder}")
        return False

    logger.info(f"Downloading {photo_set.folder} (~{photo_set.approx_mb} MB) "
                f"from {photo_set.repo}")
    try:
        response = requests.get(photo_set.zip_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Download failed for {photo_set.folder}: {exc}")
        return False

    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            count = _extract_images(archive, target)
    except zipfile.BadZipFile as exc:
        logger.error(f"Bad archive for {photo_set.folder}: {exc}")
        return False

    if count == 0:
        logger.error(f"No images found inside {photo_set.repo}")
        return False

    with open(target / "SOURCE.json", "w", encoding="utf-8") as handle:
        json.dump({
            "folder": photo_set.folder,
            "repository": photo_set.repo_url,
            "branch": photo_set.branch,
            "licence": photo_set.licence,
            "note": photo_set.note,
            "image_count": count,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, handle, indent=2)

    logger.success(f"{photo_set.folder}: {count} images → {target}")
    logger.info(f"  Reconstruct with: python scripts/run_meshroom.py "
                f"--object {photo_set.folder}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch open photogrammetry image sets for Meshroom testing",
    )
    parser.add_argument(
        "--set", action="append", choices=sorted(SETS_BY_KEY),
        help="Set to fetch (repeatable)",
    )
    parser.add_argument("--all", action="store_true", help="Fetch every set")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download sets that already exist")
    parser.add_argument("--list", action="store_true",
                        help="List available sets and exit")
    args = parser.parse_args()

    if args.list or not (args.set or args.all):
        logger.info("Available image sets:")
        for photo_set in PHOTO_SETS:
            target = SCAN_PHOTOS_DIR / photo_set.folder
            # Deduplicate by resolved path: NTFS glob is case-insensitive, so
            # globbing "*.jpg" and "*.JPG" separately double-counts on Windows.
            have = len({
                path.resolve() for path in target.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }) if target.exists() else 0
            state = f"{have} images local" if have else "not fetched"
            logger.info(f"  {photo_set.key:<10} {photo_set.folder:<16} "
                        f"~{photo_set.approx_mb:>4} MB  [{state}]")
            logger.info(f"             {photo_set.note}")
        if args.list:
            return 0
        logger.info("Pass --set <key> or --all to download.")
        return 0

    chosen = PHOTO_SETS if args.all else [SETS_BY_KEY[key] for key in args.set]
    total_mb = sum(photo_set.approx_mb for photo_set in chosen)
    logger.info(f"{len(chosen)} set(s), ~{total_mb} MB total")

    fetched = sum(fetch_set(photo_set, args.overwrite) for photo_set in chosen)
    logger.success(f"Fetched {fetched}/{len(chosen)} set(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
