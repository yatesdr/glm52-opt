#!/usr/bin/env python3
"""Release smoke for the v20 dynamickv+autocal compose: prefill bench,
decode bench, and a deep needle retrieval at a target token depth.
Stdlib only; run on the serving host: python3 v20_release_smoke.py
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.request

BASE = "http://localhost:5001"
MODEL = "GLM-5.2"
NEEDLE = "The maintenance ticket number is 738216."
QUESTION = (
    "\n\nQuestion: What is the maintenance ticket number mentioned above? "
    "Answer with only the number.\nAnswer:"
)

_WORDS = (
    "system pipeline latency register cluster thermal packet vector cache "
    "schedule replica throughput barrier index tensor kernel stream buffer "
    "fabric module policy quorum ledger metric window socket driver queue "
    "shard segment router uplink payload batch epoch checkpoint gradient"
).split()


def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _filler(n_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    total = 0
    while total < n_chars:
        sent = " ".join(rng.choice(_WORDS) for _ in range(12)).capitalize() + "."
        out.append(sent)
        total += len(sent) + 1
    return " ".join(out)


def _tokens_of(text: str) -> int:
    return len(_post("/tokenize", {"model": MODEL, "prompt": text}, 120)["tokens"])


def _sized_prompt(target_tokens: int, seed: int) -> tuple[str, float]:
    sample = _filler(20000, seed)
    cpt = len(sample) / _tokens_of(sample)  # chars per token, measured
    body = _filler(int(target_tokens * cpt * 0.985), seed + 1)
    return body, cpt


def completion(prompt: str, max_tokens: int, timeout: float) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = _post(
        "/v1/completions",
        {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0},
        timeout,
    )
    return r, time.perf_counter() - t0


def main() -> int:
    print("== v20 release smoke ==", flush=True)

    # 1) Prefill bench: cold ~64k-token prompt, 1 output token.
    body, cpt = _sized_prompt(64000, seed=101)
    r, dt = completion(body + "\nSummary word:", 1, 1800)
    ptok = r["usage"]["prompt_tokens"]
    print(f"prefill: {ptok} tokens in {dt:.2f}s -> {ptok/dt:,.0f} tok/s "
          f"(chars/token={cpt:.2f})", flush=True)

    # 2) Decode bench: tiny prompt, 512 tokens, timed twice (2nd = warmed).
    for label in ("decode-warmup", "decode"):
        r, dt = completion("Count slowly and describe each number: one,", 512, 600)
        ctok = r["usage"]["completion_tokens"]
        print(f"{label}: {ctok} tokens in {dt:.2f}s -> {ctok/dt:.2f} tok/s",
              flush=True)

    # 3) Needle at target depth: needle at 40% of a ~350k-token context.
    target = 350_000
    body, _ = _sized_prompt(target, seed=202)
    cut = int(len(body) * 0.4)
    prompt = body[:cut] + "\n" + NEEDLE + "\n" + body[cut:] + QUESTION
    r, dt = completion(prompt, 24, 3600)
    ptok = r["usage"]["prompt_tokens"]
    text = r["choices"][0]["text"].strip()
    ok = "738216" in text
    print(f"needle: prompt_tokens={ptok} elapsed={dt:.1f}s "
          f"answer={text!r} verdict={'EXACT' if ok else 'MISS'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
