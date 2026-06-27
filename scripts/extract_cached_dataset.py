"""
AXUM ROVER — Extract HHD-Ethiopic from HuggingFace cache
=========================================================
Reads the already-downloaded dataset blob directly,
bypassing the broken HuggingFace loading script.

Run: python scripts/extract_cached_dataset.py
"""

import io
import struct
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa

# ── Config ────────────────────────────────────────────────────────────────────

BLOB_DIR   = Path.home() / ".cache/huggingface/hub/datasets--OCR-Ethiopic--HHD-Ethiopic/blobs"
OUTPUT_DIR = Path("data/geez_characters")

# ── Find blob files ───────────────────────────────────────────────────────────

def find_blobs():
    if not BLOB_DIR.exists():
        print(f"ERROR: Blob directory not found: {BLOB_DIR}")
        print("Check the path manually and update BLOB_DIR above.")
        return []

    blobs = sorted(BLOB_DIR.iterdir())
    print(f"Found {len(blobs)} blob file(s) in cache:")
    for b in blobs:
        mb = b.stat().st_size / 1024 / 1024
        print(f"  {b.name[:40]}...  {mb:.1f} MB")
    return blobs


# ── Try reading as Parquet ────────────────────────────────────────────────────

def try_parquet(blob_path: Path) -> bool:
    """Try to read blob as a parquet file and extract images."""
    print(f"\nTrying parquet read on: {blob_path.name[:50]}...")
    try:
        table = pq.read_table(str(blob_path))
        print(f"  ✓ Parquet opened — columns: {table.column_names}")
        print(f"  ✓ Rows: {len(table)}")
        extract_from_table(table)
        return True
    except Exception as e:
        print(f"  ✗ Not a parquet file: {e}")
        return False


def extract_from_table(table: pa.Table):
    """Extract images and labels from an Arrow table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = table.column_names
    print(f"  Columns: {cols}")

    # Find image and label columns — dataset may use different names
    img_col   = next((c for c in cols if 'image' in c.lower()), None)
    label_col = next((c for c in cols if c.lower() in
                      ('text', 'label', 'character', 'char', 'word')), None)

    if not img_col:
        print(f"  ✗ No image column found in {cols}")
        return
    if not label_col:
        print(f"  ✗ No label column found in {cols}")
        return

    print(f"  Using image col: '{img_col}', label col: '{label_col}'")

    saved   = 0
    skipped = 0

    for i in range(len(table)):
        row        = table.slice(i, 1)
        label_val  = row[label_col][0].as_py()
        image_val  = row[img_col][0].as_py()

        if not label_val:
            skipped += 1
            continue

        # Label: use first character as folder name
        char = str(label_val)[0] if label_val else None
        if not char:
            skipped += 1
            continue

        char_dir = OUTPUT_DIR / char
        char_dir.mkdir(exist_ok=True)

        # Image value can be: dict with 'bytes', raw bytes, or PIL image
        try:
            if isinstance(image_val, dict):
                img_bytes = image_val.get('bytes') or image_val.get('path')
                if isinstance(img_bytes, bytes):
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                else:
                    skipped += 1
                    continue
            elif isinstance(image_val, bytes):
                img = Image.open(io.BytesIO(image_val)).convert('RGB')
            elif hasattr(image_val, 'save'):
                img = image_val
            else:
                skipped += 1
                continue

            out_path = char_dir / f"{i:07d}.png"
            img.save(out_path)
            saved += 1

            if saved % 500 == 0:
                print(f"  Saved {saved} images...")

        except Exception as e:
            skipped += 1
            if skipped < 5:
                print(f"  Warning on row {i}: {e}")

    print(f"\n✓ Extraction complete: {saved} saved, {skipped} skipped")
    print(f"  Output: {OUTPUT_DIR.resolve()}")


# ── Try reading as Arrow IPC stream ──────────────────────────────────────────

def try_arrow_ipc(blob_path: Path) -> bool:
    """Try reading as Arrow IPC (feather) format."""
    print(f"\nTrying Arrow IPC read...")
    try:
        import pyarrow.ipc as ipc
        with open(blob_path, 'rb') as f:
            reader = ipc.open_stream(f)
            table  = reader.read_all()
        print(f"  ✓ Arrow IPC opened — {len(table)} rows")
        extract_from_table(table)
        return True
    except Exception as e:
        print(f"  ✗ Not Arrow IPC: {e}")
        return False


def try_arrow_file(blob_path: Path) -> bool:
    """Try reading as Arrow file format."""
    print(f"\nTrying Arrow file read...")
    try:
        import pyarrow.ipc as ipc
        with open(blob_path, 'rb') as f:
            reader = ipc.open_file(f)
            table  = reader.read_all()
        print(f"  ✓ Arrow file opened — {len(table)} rows")
        extract_from_table(table)
        return True
    except Exception as e:
        print(f"  ✗ Not Arrow file: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("AXUM — HHD-Ethiopic Cache Extractor\n")

    # Check pyarrow is available
    try:
        import pyarrow
        print(f"✓ pyarrow {pyarrow.__version__} available")
    except ImportError:
        print("✗ pyarrow not installed — run: pip install pyarrow")
        exit(1)

    blobs = find_blobs()
    if not blobs:
        exit(1)

    # Try each blob with each format
    success = False
    for blob in blobs:
        mb = blob.stat().st_size / 1024 / 1024
        if mb < 1:
            print(f"\nSkipping small file ({mb:.1f}MB): {blob.name[:40]}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {blob.name[:50]} ({mb:.1f} MB)")
        print('='*60)

        if try_parquet(blob):
            success = True
            break
        if try_arrow_ipc(blob):
            success = True
            break
        if try_arrow_file(blob):
            success = True
            break

        print(f"  ✗ Could not read {blob.name[:40]} in any known format")

    if success:
        print("\n" + "="*60)
        print("SUCCESS — Now verify the extraction:")
        print("  python -c \"from src.ocr.pipeline import verify_dataset; verify_dataset('data/geez_characters')\"")
        print("\nThen start training:")
        print("  python scripts/train_ocr.py")
    else:
        print("\n" + "="*60)
        print("All formats failed. The blob may be a ZIP or tar archive.")
        print("Run this to check the file header:")
        print(f"  Format-Hex '{blobs[0]}' | Select-Object -First 4")
        print("Paste the output and we'll handle the format.")