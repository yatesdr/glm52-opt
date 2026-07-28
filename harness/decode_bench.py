#!/usr/bin/env python3
"""Measure aggregate decode throughput at several concurrency levels.

Requests use short, unique prompts and ``ignore_eos`` so every cell performs
the same amount of decode work. Results include client and server token
counters, per-user throughput, interval overlap, and MTP acceptance deltas.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
import urllib.request


def http(url: str, payload: dict | None = None, timeout: int = 600) -> str:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def scrape(base: str) -> dict[str, float]:
    wanted = (
        "vllm:generation_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    values = {name: 0.0 for name in wanted}
    for line in http(f"{base}/metrics").splitlines():
        for name in wanted:
            if line.startswith(name + "{"):
                values[name] += float(line.rsplit(" ", 1)[1])
    return values


FILLER = (
    "The consolidation entries for the period were reviewed by the plant "
    "controllers before submission. Intercompany balances cleared without "
    "manual adjustment, and the hedging position rolled forward unchanged. "
)


def context_prefix(context_tokens: int, per_block: int = 34) -> str:
    """Natural-language padding so decode can be measured at a non-zero context
    (Gate 3 asks for ctx0 AND ctx16k; only ctx0 had ever been run)."""
    if context_tokens <= 0:
        return ""
    blocks = max(1, context_tokens // per_block)
    return ("Background packet for this review:\n" + FILLER * blocks +
            "\n\nWith that context in mind: ")


def run_one(
    base: str,
    model: str,
    output_tokens: int,
    index: int,
    barrier: threading.Barrier,
    context_tokens: int = 0,
    nonce_prefix: str | None = None,
) -> dict:
    nonce = (
        os.urandom(8).hex()
        if nonce_prefix is None
        else f"{nonce_prefix}-{index}"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    context_prefix(context_tokens) +
                    f"Request {index}, nonce {nonce}. Write a long, detailed "
                    "numbered account of building a wooden chair. Continue "
                    "until the response limit."
                ),
            }
        ],
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    barrier.wait()
    started = time.perf_counter()
    response = json.loads(
        http(f"{base}/v1/chat/completions", payload, timeout=900)
    )
    ended = time.perf_counter()
    choice = response["choices"][0]
    message = choice["message"]
    completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
    return {
        "started": started,
        "ended": ended,
        "prompt_tokens": int(response.get("usage", {}).get("prompt_tokens", 0)),
        "cached_tokens": int((response.get("usage", {}).get("prompt_tokens_details") or {})
                            .get("cached_tokens") or 0),
        "duration_s": ended - started,
        "completion_tokens": completion_tokens,
        "finish_reason": choice.get("finish_reason"),
        "finalized": bool(
            (message.get("content") or "").strip()
            or (message.get("reasoning") or "").strip()
            or (message.get("reasoning_content") or "").strip()
        ),
    }


def max_overlap(results: list[dict]) -> int:
    events: list[tuple[float, int]] = []
    for result in results:
        events.append((result["started"], 1))
        events.append((result["ended"], -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:5001")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--context-tokens", type=int, default=0, dest="context_tokens",
                        help="prefix each request with ~N tokens of natural filler (Gate 3 ctx16k)")
    parser.add_argument(
        "--nonce-prefix",
        default=None,
        help=(
            "use a deterministic request nonce prefix; the request index is "
            "appended so matched image/config comparisons receive identical prompts"
        ),
    )
    args = parser.parse_args()

    for concurrency in (
        int(value) for value in args.concurrency.split(",") if value
    ):
        before = scrape(args.base)
        barrier = threading.Barrier(concurrency + 1)
        results: list[dict] = []
        errors: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = [
                executor.submit(
                    run_one,
                    args.base,
                    args.model,
                    args.output_tokens,
                    index,
                    barrier,
                    args.context_tokens,
                    args.nonce_prefix,
                )
                for index in range(concurrency)
            ]
            barrier.wait()
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:  # Preserve all failures in the result.
                    errors.append(repr(error))
        after = scrape(args.base)

        if results:
            elapsed = max(r["ended"] for r in results) - min(
                r["started"] for r in results
            )
            client_tokens = sum(r["completion_tokens"] for r in results)
            per_user = [
                r["completion_tokens"] / r["duration_s"] for r in results
            ]
        else:
            elapsed = 0.0
            client_tokens = 0
            per_user = []
        server_tokens = (
            after["vllm:generation_tokens_total"]
            - before["vllm:generation_tokens_total"]
        )
        draft_tokens = (
            after["vllm:spec_decode_num_draft_tokens_total"]
            - before["vllm:spec_decode_num_draft_tokens_total"]
        )
        accepted_tokens = (
            after["vllm:spec_decode_num_accepted_tokens_total"]
            - before["vllm:spec_decode_num_accepted_tokens_total"]
        )
        print(
            json.dumps(
                {
                    "concurrency": concurrency,
                    "requests_ok": len(results),
                    "errors": errors,
                    "output_tokens_requested_each": args.output_tokens,
                    "client_completion_tokens": client_tokens,
                    "server_generation_tokens": int(server_tokens),
                    "elapsed_s": round(elapsed, 3),
                    "aggregate_tok_s": round(
                        server_tokens / elapsed if elapsed else 0.0, 2
                    ),
                    "median_per_user_tok_s": round(
                        statistics.median(per_user) if per_user else 0.0, 2
                    ),
                    "effective_peak_concurrency": max_overlap(results),
                    "finish_reasons": sorted(
                        {str(r["finish_reason"]) for r in results}
                    ),
                    "all_finalized": all(r["finalized"] for r in results),
                    "mtp_draft_tokens": int(draft_tokens),
                    "mtp_accepted_tokens": int(accepted_tokens),
                    "mtp_acceptance": round(
                        accepted_tokens / draft_tokens if draft_tokens else 0.0,
                        4,
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
