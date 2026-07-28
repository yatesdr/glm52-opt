#!/usr/bin/env python3
"""Prove the NVFP4 writer semantics from a root-owned SparkInfer cache."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys


def strings_in(value: object):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2].lower() not in {"true", "false"}:
        print(f"usage: {sys.argv[0]} CACHE_ROOT {{true|false}}", file=sys.stderr)
        return 2
    root = pathlib.Path(sys.argv[1])
    expected = sys.argv[2].lower() == "true"
    writer_specs = []
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        text = "\n".join(strings_in(payload))
        if '"kernel":"attention.mla.nvfp4_fp8_rope_kv_cache"' not in text:
            continue
        dynamic = '"per_token_scale",true' in text
        static = '"per_token_scale",false' in text
        if dynamic == static:
            continue
        writer_specs.append(
            {
                "path": str(path),
                "per_token_scale": dynamic,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    matching = [row for row in writer_specs if row["per_token_scale"] is expected]
    proof = {
        "expected_per_token_scale": expected,
        "writer_specs": writer_specs,
        "matching_writer_specs": len(matching),
        "all_writer_specs_match_expected": bool(writer_specs)
        and all(row["per_token_scale"] is expected for row in writer_specs),
    }
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0 if proof["all_writer_specs_match_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
