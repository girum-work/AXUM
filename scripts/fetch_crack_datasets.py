"""
fetch_crack_datasets.py — assemble labelled crack imagery for segmentation.

The OpenCV detector scores F1 0.111 on Stone331 and its severity barely ranks
damage (rank correlation +0.16). Training a segmenter needs labelled masks, and
the repository has none of its own.

MCS is fetched automatically: it lives inside a public git repository and is
CC BY 4.0, the only crack dataset located so far that may be redistributed.

Stone331 is not fetched. DeepCrack hosts it behind OneDrive folder links that
cannot be scripted, and the repository declares no licence at all, so it is
research-use only and must be placed by hand.

    MCS       246 marble surfaces   256x256    CC BY 4.0
    Stone331  331 stone surfaces    1024x1024  no licence declared

DO NOT MERGE THESE NAIVELY. They follow different annotation conventions,
measured over 120 masks each:

    MCS       crack width 21.9px on 256px   8.56% of frame   5.34% of pixels
    Stone331  crack width  1.9px on 512px   0.37% of frame   0.11% of pixels

MCS labels the crack BODY, Stone331 traces a thin CENTRELINE -- a 23x
difference in width. A model trained on both learns two contradictory targets.
Train per dataset, or harmonise first by skeletonising MCS or dilating
Stone331. Which convention is right depends on the use: body width carries
severity information that a centreline discards.

MCS masks are paletted PNGs, and reading them wrongly fails silently:

    PIL, mode P        palette index   {0, 1}      correct
    cv2 GRAYSCALE      {0, 38}                     correct after > 0
    cv2 UNCHANGED      (H, W, 3), crack in ch 2    channel 0 is ALL ZEROS

Taking channel 0 of an UNCHANGED read yields empty masks, so training appears
to run normally while learning nothing.

A note on MCS worth reading before trusting a model trained on it: marble
commonly contains fissures -- cracks naturally refilled with minerals, which
look like damage but do not threaten stability. The dataset authors raise this
explicitly. Whether the masks label fissures as crack decides whether a model
learns "visually crack-like" or "structurally damaged", and those are different
questions. Run --inspect to see what the masks actually contain.

LARGER SOURCES. 577 images is the binding constraint on this work, and it is
self-imposed: the field has assembled far larger collections.

VERIFIED AND USABLE. GT-CrackSeg, the codebase released alongside the 2025
review (arXiv 2508.10256), publishes four datasets from one Drive folder linked
in its README, with the instruction "please follow each dataset's original
license/terms":

    https://github.com/GT-SCI/GT-CrackSeg

    Khanhha    ~11,200 images, merged from 12 crack segmentation datasets
    TUT          1,408 images, diverse real-world backgrounds
    DeepCrack      537 images, the family Stone331 belongs to
    CrackTree      206 images

Khanhha alone is 19x what we train on. The download is a Drive FOLDER, which is
not scriptable for the same reason Stone331's OneDrive link is not, so it is
fetched by hand. Unpack into data/crack_datasets/<name>/<Split>/{img,anno} and
scripts/train_cracknet.py finds it automatically -- that is GT-CrackSeg's own
layout, kept deliberately so their configs and ours agree.

Their loader carries this comment, which is worth knowing before resizing
anything: "Skip resizing for CrackTree (thin cracks degrade with resizing)".
That is the same effect measured here, where downscaling Stone331 512->256
shattered one connected centreline into ~29 fragments.

NOT YET VERIFIED. Sizes are as reported by the authors; licences unchecked and
no links recorded because none were confirmed:

    OmniCrack30k   30,000 images, 9 billion pixels, 20+ datasets, spanning
                   asphalt, ceramic, concrete, masonry and steel. CVPR W 2024.
    CrackSeg9k     ~9,000 images. Combines earlier datasets AND unifies their
                   annotations, which is the MCS-vs-Stone331 convention problem
                   solved properly rather than by our skeletonising workaround.
    Conglomerate   10,995 images from Virginia DOT bridge inspection reports.
    S2DS           743 images labelled crack, spalling, corrosion,
                   efflorescence, VEGETATION and control point. The vegetation
                   class matters here: lichen is what the filter pipeline kept
                   marking as crack, and this labels it as a separate thing.
    Masonry        Photos of masonry structures with complex backgrounds --
                   closer to Aksumite stelae than concrete or marble is.

Syncrack (VISAPP 2022) is worth reading before trusting any of them: it shows
manual crack annotations are systematically inaccurate at pixel level, and that
the resulting learning bias measurably hinders pixel-accurate detection. That is
the same effect measured here on Stone331, where the stone's own texture
out-responds its annotated cracks (median ridge response inside a labelled crack
0.057, against a 99th percentile of 0.102 on crack-free area).

Usage:
    python scripts/fetch_crack_datasets.py --dataset mcs
    python scripts/fetch_crack_datasets.py --inspect
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests

MCS_ZIP = ("https://codeload.github.com/MachineLearningVisionRG/"
           "mcs-dataset/zip/refs/heads/main")
MCS_HOME = "https://github.com/MachineLearningVisionRG/mcs-dataset"
USER_AGENT = "AXUM-Rover-Crack/1.0 (heritage conservation research)"

DATA_ROOT = Path("data/crack_datasets")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def fetch_mcs(destination: Path) -> int:
    """
    Download the Marble Crack Segmentation dataset.

    Args:
        destination: Root to write images/ and masks/ under

    Returns:
        Process exit code
    """
    images_dir = destination / "images"
    masks_dir = destination / "masks"
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"MCS already present at {destination}")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    print(f"Downloading {MCS_ZIP}")
    try:
        response = session.get(MCS_ZIP, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "masks": 0}
    for member in archive.namelist():
        if member.endswith("/"):
            continue
        parts = member.split("/")
        if "dataset" not in parts:
            continue
        index = parts.index("dataset")
        if len(parts) < index + 3:
            continue
        kind = parts[index + 1]
        if kind == "images":
            (images_dir / parts[-1]).write_bytes(archive.read(member))
            counts["images"] += 1
        elif kind == "masks":
            (masks_dir / parts[-1]).write_bytes(archive.read(member))
            counts["masks"] += 1

    (destination / "LICENCE.txt").write_text(
        f"Marble Crack Segmentation (MCS) dataset\n"
        f"Source: {MCS_HOME}\nLicence: CC BY 4.0 -- attribution required.\n",
        encoding="utf-8",
    )
    print(f"  images {counts['images']}, masks {counts['masks']}")
    return 0 if counts["images"] else 1


def inspect(root: Path) -> int:
    """Report pairing, size and how much of each mask is marked crack."""
    found = False
    for dataset in sorted(p for p in root.parent.rglob("*") if p.is_dir()):
        images_dir = dataset / "images"
        masks_dir = next((dataset / n for n in ("masks", "gt")
                          if (dataset / n).exists()), None)
        if not images_dir.exists() or masks_dir is None:
            continue
        found = True

        images = {p.stem: p for p in images_dir.iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES}
        masks = {p.stem: p for p in masks_dir.iterdir()
                 if p.suffix.lower() in IMAGE_SUFFIXES}
        shared = sorted(set(images) & set(masks))

        fractions, sizes, values = [], set(), set()
        for stem in shared[:120]:
            mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(images[stem]))
            if mask is None or image is None:
                continue
            fractions.append(np.count_nonzero(mask > 0) / mask.size)
            sizes.add((image.shape[1], image.shape[0]))
            values.update(np.unique(mask).tolist()[:6])

        print(f"\n{dataset.relative_to(root.parent)}")
        print(f"  paired          : {len(shared)} "
              f"({len(images)} images, {len(masks)} masks)")
        print(f"  image sizes     : {sorted(sizes)[:4]}")
        print(f"  mask values     : {sorted(values)[:8]}")
        if fractions:
            print(f"  crack pixels    : mean {100 * np.mean(fractions):.3f}%  "
                  f"range {100 * min(fractions):.3f}-{100 * max(fractions):.3f}%")
            empty = sum(1 for f in fractions if f == 0)
            if empty:
                print(f"  empty masks     : {empty}/{len(fractions)} "
                      f"(images with no crack labelled)")

    if not found:
        print(f"No image/mask pairs found under {root.parent}", file=sys.stderr)
        return 1
    return 0


def import_bundle(archive: Path, root: Path) -> int:
    """
    Unpack the GT-CrackSeg bundle, which is a zip of per-dataset zips.

    Zipped on macOS, so every archive carries a parallel __MACOSX/ tree of
    AppleDouble resource forks named ._<file>. Those are not images, but they
    sit next to the real ones and index_by_stem would happily pair them, so
    they are dropped rather than extracted.
    """
    if not archive.exists():
        print(f"No archive at {archive}", file=sys.stderr)
        return 1

    root.mkdir(parents=True, exist_ok=True)
    staging = root / "_staging"
    with zipfile.ZipFile(archive) as outer:
        outer.extractall(staging)

    inner_archives = sorted(staging.rglob("*.zip"))
    if not inner_archives:
        print(f"No inner archives inside {archive.name}", file=sys.stderr)
        return 1

    for inner in inner_archives:
        name = inner.stem.lower()
        destination = root / name
        written = 0
        with zipfile.ZipFile(inner) as zf:
            for entry in zf.namelist():
                if entry.endswith("/") or "__MACOSX" in entry:
                    continue
                if Path(entry).name.startswith(("._", ".DS_Store")):
                    continue
                # Drop the archive's own top-level folder so the dataset name
                # is ours, not whatever case the zip happened to use.
                relative = Path(*Path(entry).parts[1:])
                if not relative.parts:
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(entry) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                written += 1
        print(f"  {name:12s} {written:>7,d} files -> {destination}")

    shutil.rmtree(staging, ignore_errors=True)
    print(f"\nUnpacked into {root}. scripts/train_cracknet.py discovers the\n"
          f"<name>/<Split>/{{img,anno}} layout automatically.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcs"], help="Dataset to download")
    parser.add_argument("--inspect", action="store_true",
                        help="Report what the downloaded masks contain")
    parser.add_argument("--import-bundle", type=Path,
                        help="Unpack the GT-CrackSeg zip-of-zips download")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    if args.import_bundle:
        return import_bundle(args.import_bundle, args.root)
    if args.inspect:
        return inspect(args.root / "mcs")
    if args.dataset == "mcs":
        return fetch_mcs(args.root / "mcs")

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
