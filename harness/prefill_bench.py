#!/usr/bin/env python3
"""Prefill throughput bench against a local vLLM server.

Measures server-side via /metrics deltas (request_prefill_time_seconds_sum /
prompt_tokens_total), which is immune to client/network jitter, plus wall time.
Each run uses a unique prefix so prefix caching cannot contaminate the number.

Usage:
  python3 prefill_bench.py --tokens 8000
  python3 prefill_bench.py --tokens 55000 --label fp8ring
"""

import argparse
import json
import os
import time
import urllib.request

FILLER = (
    "Plant output for the period held steady while material costs drifted "
    "upward against the quarterly plan. The operations team reviewed scrap "
    "rates, tooling downtime, and shift coverage across the affected lines. "
    "Freight recovery lagged the index by a small margin, and the variance "
    "was attributed to carrier mix rather than volume. Working capital "
    "remained inside the corridor agreed with the treasury group. "
)


def http(url, payload=None, timeout=600):
    if payload is not None:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def count_tokens(base, model, text):
    out = json.loads(http(f"{base}/tokenize", {"model": model, "prompt": text}))
    return out["count"]


def scrape(base):
    vals = {}
    for line in http(f"{base}/metrics").splitlines():
        for key in (
            "vllm:prompt_tokens_total",
            "vllm:request_prefill_time_seconds_sum",
            "vllm:request_prefill_time_seconds_count",
        ):
            if line.startswith(key + "{"):
                vals[key] = float(line.rsplit(" ", 1)[1])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    # Build a prompt of ~target tokens, unique prefix defeats prefix cache.
    uid = os.urandom(8).hex()
    per_block = count_tokens(args.base, args.model, FILLER)
    blocks = max(1, args.tokens // per_block)
    prompt = f"Report id {uid}.\n" + FILLER * blocks
    n = count_tokens(args.base, args.model, prompt)
    # trim/pad to within ~2%
    while n > args.tokens * 1.02 and blocks > 1:
        blocks -= max(1, (n - args.tokens) // per_block)
        prompt = f"Report id {uid}.\n" + FILLER * blocks
        n = count_tokens(args.base, args.model, prompt)

    before = scrape(args.base)
    t0 = time.time()
    http(
        f"{args.base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                    + "\n\nReply with the single word: acknowledged.",
                }
            ],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
    )
    wall = time.time() - t0
    after = scrape(args.base)

    dtok = after["vllm:prompt_tokens_total"] - before["vllm:prompt_tokens_total"]
    dt = (
        after["vllm:request_prefill_time_seconds_sum"]
        - before["vllm:request_prefill_time_seconds_sum"]
    )
    server_tps = dtok / dt if dt > 0 else float("nan")
    print(
        f"RESULT label={args.label or 'run'} prompt_tokens={int(dtok)} "
        f"prefill_time_s={dt:.2f} server_prefill_tok_s={server_tps:.0f} "
        f"wall_s={wall:.2f} wall_tok_s={n / wall:.0f}"
    )


if __name__ == "__main__":
    main()
