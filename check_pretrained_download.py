"""
AXUM ROVER - Isolated pretrained weight download check.

WHAT: Tests ONLY whether real EfficientNet-B3 pretrained weights can be
downloaded and loaded, isolated from everything else in the classifier
pipeline. Doesn't test device tiering, forward-pass shape, or anything
else already logic-verified in Systems Integration Engineer's sandbox.

WHY isolated: Systems Integration Engineer's sandbox can't reach
download.pytorch.org at all (confirmed connection failure, not a slow
download) and has substituted pretrained=False to test everything else.
That leaves exactly one real unknown before tomorrow: does the download
+ load actually work on a machine with normal internet access. This
script answers only that question, in under a minute, without needing
to run the full classifier or touch any AXUM-specific code.

USAGE:
    python check_pretrained_download.py
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    print("Checking real pretrained EfficientNet-B3 weight download...")
    print("(This is the ONE piece nobody has verified yet before tomorrow's session.)\n")

    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"FAIL: torch/torchvision not installed here: {exc}")
        print("This machine can't even test this — not a network problem, an environment problem.")
        return 1

    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    print()

    start = time.monotonic()
    try:
        model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"FAIL after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        print("\nThis is the real blocker — if this fails here, it will fail identically tomorrow")
        print("at the studio unless that machine has different network access.")
        return 1

    elapsed = time.monotonic() - start
    param_count = sum(p.numel() for p in model.parameters())
    print(f"SUCCESS in {elapsed:.1f}s")
    print(f"Model loaded: {param_count:,} parameters")

    # Sanity check it's actually usable, not just downloaded - a real forward pass
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 300, 300)  # EfficientNet-B3's expected input size
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Forward pass sane: output shape {tuple(output.shape)} (expect (1, 1000) for ImageNet head)")
    except Exception as exc:
        print(f"WARNING: weights loaded but forward pass failed: {exc}")
        print("Download works, but something else is wrong — worth investigating before tomorrow too.")
        return 1

    print("\nReal pretrained weight download + load + forward pass: all confirmed working on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())