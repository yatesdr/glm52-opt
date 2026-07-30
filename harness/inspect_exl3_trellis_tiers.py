#!/usr/bin/env python3
"""Inspect EXL3 per-expert Trellis widths without loading tensor payloads."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=pathlib.Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    index_path = args.model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    pattern = re.compile(
        rf"^model\.layers\.{args.layer}\.mlp\.experts\.(\d+)\."
        rf"(gate_proj|up_proj|down_proj)\.rank{args.rank}\.trellis$"
    )
    by_file: dict[str, list[tuple[str, int, str]]] = collections.defaultdict(list)
    for key, filename in weight_map.items():
        match = pattern.match(key)
        if match:
            by_file[filename].append((key, int(match.group(1)), match.group(2)))

    widths: dict[int, set[int]] = collections.defaultdict(set)
    for filename, entries in by_file.items():
        with safe_open(args.model_dir / filename, framework="pt", device="cpu") as handle:
            for key, expert, _projection in entries:
                shape = handle.get_slice(key).get_shape()
                widths[expert].add(int(shape[-1]) // 16)

    inconsistent = {expert: values for expert, values in widths.items() if len(values) != 1}
    if inconsistent:
        raise SystemExit(f"inconsistent projection widths: {inconsistent}")

    tiers: dict[int, list[int]] = collections.defaultdict(list)
    for expert, values in sorted(widths.items()):
        tiers[next(iter(values))].append(expert)

    print(f"layer={args.layer} rank={args.rank} experts={len(widths)}")
    for bits, experts in sorted(tiers.items()):
        contiguous = experts == list(range(experts[0], experts[-1] + 1))
        print(
            f"K={bits} count={len(experts)} first={experts[0]} "
            f"last={experts[-1]} contiguous={str(contiguous).lower()}"
        )


if __name__ == "__main__":
    main()
