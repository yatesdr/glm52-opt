#!/usr/bin/env python3
"""Map frozen rendered token positions to tokenizer text via the API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:5001")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    token_ids = json.loads(args.ids.read_text())
    if not isinstance(token_ids, list):
        raise TypeError("rendered token IDs must be a JSON list")
    if not 0 <= args.start < args.end <= len(token_ids):
        raise ValueError("requested position window is out of range")

    rows = []
    for position in range(args.start, args.end):
        token_id = int(token_ids[position])
        payload = json.dumps(
            {"model": args.model, "tokens": [token_id]}
        ).encode()
        request = urllib.request.Request(
            args.base + "/detokenize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        rows.append(
            {
                "position": position,
                "token_id": token_id,
                "text": result["prompt"],
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
