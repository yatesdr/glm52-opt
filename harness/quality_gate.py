#!/usr/bin/env python3
"""Quality gate for comm-mode experiments (fp8 ring/a2a wire compression).

Three checks, exit 0 only if all pass:
  1. needle retrieval at ~55k tokens (benign phrasing — the model treats
     "secret code" style needles as injection attempts)
  2. multi-step arithmetic, exact answer
  3. long-generation coherence: 150+ words of FP&A prose, sanity-checked
     for repetition loops (the failure mode quantized comms tend to show)

Usage: python3 quality_gate.py [--base http://localhost:5001] [--depth-tokens 55000]
"""

import argparse
import json
import re
import sys
import urllib.request

FILLER = (
    "The consolidation entries for the period were reviewed by the plant "
    "controllers before submission. Intercompany balances cleared without "
    "manual adjustment, and the hedging position rolled forward unchanged. "
    "Tooling amortization followed the agreed schedule, while indirect "
    "spend stayed within the corridor set at the last quarterly review. "
)

NEEDLE = (
    "For reference, the maintenance ticket number for the Facility 27 "
    "compressor overhaul is 738216. "
)


def chat(base, content, max_tokens=1200):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "GLM-5.2",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def tokens(base, text):
    req = urllib.request.Request(
        f"{base}/tokenize",
        data=json.dumps({"model": "GLM-5.2", "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--depth-tokens", type=int, default=55000)
    args = ap.parse_args()
    ok = True

    # 1. needle at ~40% depth of ~55k tokens
    per = tokens(args.base, FILLER)
    total_blocks = max(4, args.depth_tokens // per)
    pre = int(total_blocks * 0.4)
    doc = FILLER * pre + NEEDLE + FILLER * (total_blocks - pre)
    ans = chat(
        args.base,
        doc
        + "\n\nFrom the document above: what is the maintenance ticket "
        "number for the Facility 27 compressor overhaul? Reply with the "
        "number only.",
        max_tokens=600,
    )
    passed = "738216" in ans.replace(",", "")
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] needle@{args.depth_tokens}: {ans.strip()[:80]!r}")

    # 2. arithmetic: 137*89 + 4521/3 - 256 = 12193 + 1507 - 256 = 13444
    ans = chat(
        args.base,
        "Compute step by step, then give the final integer on its own line: "
        "(137 * 89) + (4521 / 3) - 256",
        max_tokens=800,
    )
    passed = "13444" in ans.replace(",", "")
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] arithmetic: tail={ans.strip()[-60:]!r}")

    # 3. coherence: long generation, check length + no degenerate repetition
    ans = chat(
        args.base,
        "In about 200 words, explain to a plant manager why freight cost "
        "recovery can lag a fuel index even when volumes are flat.",
        max_tokens=1500,
    )
    words = ans.split()
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    rep = (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams))
    passed = len(words) >= 120 and rep < 0.15
    ok &= passed
    print(
        f"[{'PASS' if passed else 'FAIL'}] coherence: {len(words)} words, "
        f"trigram-repeat={rep:.2f}"
    )
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
