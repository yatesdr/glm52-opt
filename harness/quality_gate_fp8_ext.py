#!/usr/bin/env python3
"""Extended quality gate for fp8 wire modes (deep needle + JSON echo).

Stresses the two failure modes fp8 requant hops are most likely to show:
  1. DEEP needle retrieval (default ~95% depth): at depth the attention
     softmax is extremely peaked — small query/output errors shift mass
     to wrong tokens and the needle is missed.
  2. JSON echo: exact structured round-trip of nested numeric data;
     quantization noise tends to corrupt low-salience digits first.

Usage:
  python3 quality_gate_fp8_ext.py [--base http://localhost:5001]
                                  [--depth-tokens 60000] [--deep]
--deep raises depth-tokens to 200000 (ship gate, needs full-context boot).
"""

import argparse
import json
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
    "For reference, the purchase order number for the Facility 41 chiller "
    "replacement is 592847. "
)

ECHO_PAYLOAD = {
    "plant": "F27",
    "period": "2026-06",
    "lines": [
        {"acct": "5010", "amount": 182349.27, "qty": 1152},
        {"acct": "5220", "amount": -4821.03, "qty": 87},
        {"acct": "6105", "amount": 903112.4, "qty": 20458},
    ],
    "fx_rate": 1.0872,
}


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
    with urllib.request.urlopen(req, timeout=1800) as r:
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
    ap.add_argument("--depth-tokens", type=int, default=60000)
    ap.add_argument("--deep", action="store_true", help="200k ship-gate depth")
    args = ap.parse_args()
    depth = 200000 if args.deep else args.depth_tokens
    ok = True

    # 1. needle at ~95% depth
    per = tokens(args.base, FILLER)
    total_blocks = max(4, depth // per)
    pre = int(total_blocks * 0.95)
    doc = FILLER * pre + NEEDLE + FILLER * (total_blocks - pre)
    ans = chat(
        args.base,
        doc
        + "\n\nFrom the document above: what is the purchase order number "
        "for the Facility 41 chiller replacement? Reply with the number "
        "only.",
        max_tokens=600,
    )
    passed = "592847" in ans.replace(",", "")
    ok &= passed
    print(
        f"[{'PASS' if passed else 'FAIL'}] deep-needle@{depth}@95%: "
        f"{ans.strip()[:80]!r}"
    )

    # 2. JSON echo: exact structured round-trip
    ans = chat(
        args.base,
        "Return EXACTLY the following JSON, unchanged, with no commentary "
        "and no code fences:\n" + json.dumps(ECHO_PAYLOAD),
        max_tokens=800,
    )
    cleaned = ans.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`\n")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        echoed = json.loads(cleaned)
        passed = echoed == ECHO_PAYLOAD
    except Exception:
        passed = False
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] json-echo: {cleaned[:80]!r}")

    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
