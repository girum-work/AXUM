"""
AXUM ROVER — OCR post-processing for restoration pipeline
==========================================================
Maps low-confidence CTC time steps to [MISSING] tokens so LLM
restoration receives the same format as synthetic training data.
"""

from __future__ import annotations

import torch

from config import OCR_CHAR_CONF_THRESHOLD


def greedy_decode_with_missing(
    log_probs: torch.Tensor,
    idx_to_char: dict,
    blank_idx: int,
    threshold: float | None = None,
) -> tuple[str, float]:
    """
    Greedy CTC decode; emit [MISSING] when timestep confidence is low.

    Args:
        log_probs: Shape (seq_len, 1, num_classes) or (seq_len, batch, num_classes)
        idx_to_char: Index to character map
        blank_idx: CTC blank class index
        threshold: Per-step max probability floor (default from config)

    Returns:
        (text_with_missing, mean_confidence)
    """
    if threshold is None:
        threshold = OCR_CHAR_CONF_THRESHOLD

    if log_probs.dim() == 3 and log_probs.size(1) > 0:
        lp = log_probs[:, 0, :]
    else:
        lp = log_probs.squeeze(1) if log_probs.dim() == 3 else log_probs

    probs = torch.exp(lp)
    max_probs, max_indices = torch.max(probs, dim=1)

    collapsed_chars: list[str] = []
    confs: list[float] = []
    prev_idx = None

    for t in range(max_indices.size(0)):
        idx = int(max_indices[t].item())
        conf = float(max_probs[t].item())

        if idx == prev_idx:
            continue
        prev_idx = idx

        if idx == blank_idx:
            continue

        if idx not in idx_to_char:
            continue

        char = idx_to_char[idx]
        if char in ("<BLANK>", "<UNK>"):
            continue

        if conf < threshold:
            collapsed_chars.append("[MISSING]")
        else:
            collapsed_chars.append(char)
        confs.append(conf)

    text = "".join(collapsed_chars)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf
