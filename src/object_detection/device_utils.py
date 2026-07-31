"""
device_utils.py — shared CUDA detection/fallback logic for GPU-tier model loading.

WHAT: resolve_device() checks COMPUTE_TIER against actual CUDA availability and
returns the DEVICE THAT SHOULD ACTUALLY BE USED — never trust config.COMPUTE_TIER
blindly, always resolve it against real hardware first.

WHY a shared helper instead of three copies of the same check: the CUDA-availability
check and fallback-warning behavior is identical logic across classifier, YOLO, and
anywhere else that loads a torch model — this is trivial, low-risk shared code, not
a coordinated cross-team interface (the actual EXEC-8 spec explicitly says no shared
function signatures are needed between teams; this is just avoiding copy-pasting the
same six lines three times within my own domain).

Fail-safe philosophy: if COMPUTE_TIER=gpu but no CUDA device exists, fall back to
CPU-tier weights with a loud warning — never crash, never silently run GPU-sized
weights on CPU (which would just be catastrophically slow, not "safely degraded").
"""

from typing import Tuple

import torch
from loguru import logger


def resolve_device(compute_tier: str) -> Tuple[torch.device, str]:
    """
    WHAT: resolves the requested COMPUTE_TIER against real hardware.
    WHY: callers should use the RETURNED effective_tier to pick which weights
    to load — not config.COMPUTE_TIER directly — so a GPU-tier request on a
    machine with no CUDA device correctly loads CPU-tier weights instead of
    either crashing or (worse) trying to run GPU-sized weights on CPU.

    Returns: (torch.device, effective_tier) where effective_tier is "gpu" or "cpu".
    """
    if compute_tier == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"COMPUTE_TIER=gpu requested, CUDA available: {torch.cuda.get_device_name(0)}")
            return device, "gpu"
        else:
            logger.warning(
                "COMPUTE_TIER=gpu requested but no CUDA device is available — "
                "falling back to CPU-tier weights, not GPU-sized weights on CPU. "
                "This is expected on any machine without a GPU; verify hardware "
                "if this is unexpected."
            )
            return torch.device("cpu"), "cpu"

    return torch.device("cpu"), "cpu"


if __name__ == "__main__":
    # Smoke test — must run cleanly on CPU-only hardware (this is exactly the
    # Friday-morning acceptance criterion: doesn't crash without a GPU present).
    for requested in ("cpu", "gpu"):
        device, effective = resolve_device(requested)
        print(f"requested={requested!r} -> device={device}, effective_tier={effective!r}")
    print("OK — resolves cleanly regardless of hardware.")
