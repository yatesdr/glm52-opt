#!/usr/bin/env python3
"""Fail-closed CPU gate for GLM reference needle-selection tracing."""

from __future__ import annotations

import json

import torch

from vllm.model_executor.layers.glm_official_indexer import (
    _parse_trace_range,
    reference_selection_trace_summary,
)


def main() -> None:
    indices = torch.tensor([10, 99, 100, 101, 102, 133, -1], dtype=torch.int32)
    scores = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, float("-inf")])

    assert _parse_trace_range(None) is None
    assert _parse_trace_range("") is None
    assert _parse_trace_range("100:103") == (100, 103)
    for invalid in ("100", "100:", "-1:2", "3:3", "4:3"):
        try:
            _parse_trace_range(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid trace range accepted: {invalid!r}")

    hit = reference_selection_trace_summary(
        indices,
        scores,
        needle_start=100,
        needle_end=103,
        context_radius=2,
        nearest_k=5,
    )
    assert hit["valid_selected"] == 6
    assert hit["selected_min"] == 10
    assert hit["selected_max"] == 133
    assert hit["exact_hit"] is True
    assert hit["exact_indices"] == [100, 101, 102]
    assert hit["context_hit"] is True
    assert hit["context_indices"] == [99, 100, 101, 102]
    assert [row["distance"] for row in hit["nearest"][:3]] == [0, 0, 0]

    absent = reference_selection_trace_summary(
        indices,
        scores,
        needle_start=200,
        needle_end=203,
        context_radius=4,
        nearest_k=2,
    )
    assert absent["exact_hit"] is False
    assert absent["context_hit"] is False
    assert absent["nearest"][0]["index"] == 133
    assert absent["nearest"][0]["distance"] == 67

    print(
        json.dumps(
            {
                "schema": "v20-glm-reference-needle-trace-gate-v1",
                "status": "PASS",
                "hit": hit,
                "absent": absent,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
