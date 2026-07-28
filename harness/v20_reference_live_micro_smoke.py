#!/usr/bin/env python3
"""Cheap live-metadata smoke for the diagnostic GLM reference indexer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request


MODEL = "GLM-5.2"
EXPECTED = "SYSTEM READY"
FILLER = (
    "The maintenance planning team reviewed equipment availability, freight "
    "timing, inventory balances, safety actions, and the next inspection "
    "window. No exception required escalation, and the signed work packet "
    "remained aligned with the approved operating schedule. "
)


def post(base: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:5001")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt = FILLER * 12 + f"\nReply with exactly: {EXPECTED}"
    token_count = int(
        post(args.base, "/tokenize", {"model": MODEL, "prompt": prompt}, 300)[
            "count"
        ]
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # GLM may spend more than 32 tokens in the reasoning field even for an
        # exact short answer. Keep the prompt micro-sized, but allow enough
        # completion budget to require a finalized response.
        "max_tokens": 512,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    started = time.monotonic()
    response = post(args.base, "/v1/chat/completions", payload, 900)
    elapsed = round(time.monotonic() - started, 3)
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    usage = response.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    checks = {
        "prompt_is_micro_sized": 400 <= token_count <= 700,
        "finish_reason_stop": choice.get("finish_reason") == "stop",
        "content_exact": content == EXPECTED,
        "cold": cached in (0, None),
    }
    report = {
        "schema": "v20-reference-live-micro-smoke-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "prompt_tokens_preflight": token_count,
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning": message.get("reasoning") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "usage": usage,
        "max_tokens": payload["max_tokens"],
        "response": response,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
