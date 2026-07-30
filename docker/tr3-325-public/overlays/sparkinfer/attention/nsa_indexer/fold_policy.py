"""Rank-invariant policy for the paged indexer's exact two-level fold.

The parallel fold needs a candidate workspace proportional to
``q_rows * total_slices * topk``.  The configured byte budget is both the
runtime eligibility limit and the amount reserved by the scratch planner.
Keeping policy parsing here prevents the runtime and the reservation from
disagreeing.
"""

from __future__ import annotations

import os


TWO_LEVEL_SLICE_TOKENS = 16384
TWO_LEVEL_MAX_SLICES = 32
TWO_LEVEL_FOLD_MODE_ENV = "SPARKINFER_INDEXER_TWO_LEVEL_FOLD"
TWO_LEVEL_FOLD_MAX_MIB_ENV = "SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB"
TWO_LEVEL_FOLD_MAX_MIB_DEFAULT = 256


def read_two_level_fold_policy() -> tuple[str, int]:
    """Return ``(mode, reserved_candidate_bytes)`` from the environment."""

    raw_mode = os.getenv(TWO_LEVEL_FOLD_MODE_ENV, "auto").strip().lower()
    disabled_modes = {"0", "false", "off", "no"}
    forced_modes = {"1", "true", "on", "yes"}
    if raw_mode == "auto":
        mode = "auto"
    elif raw_mode in disabled_modes:
        mode = "off"
    elif raw_mode in forced_modes:
        mode = "on"
    else:
        raise ValueError(
            f"{TWO_LEVEL_FOLD_MODE_ENV} must be auto, 0/false/off/no, or "
            f"1/true/on/yes, got {raw_mode!r}"
        )

    raw_limit_mib = os.getenv(
        TWO_LEVEL_FOLD_MAX_MIB_ENV,
        str(TWO_LEVEL_FOLD_MAX_MIB_DEFAULT),
    )
    try:
        limit_mib = int(raw_limit_mib)
    except ValueError as exc:
        raise ValueError(
            f"{TWO_LEVEL_FOLD_MAX_MIB_ENV} must be a non-negative integer, "
            f"got {raw_limit_mib!r}"
        ) from exc
    if limit_mib < 0:
        raise ValueError(
            f"{TWO_LEVEL_FOLD_MAX_MIB_ENV} must be a non-negative integer, "
            f"got {raw_limit_mib!r}"
        )
    return mode, limit_mib * 1024 * 1024

