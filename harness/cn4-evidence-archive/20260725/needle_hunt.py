#!/usr/bin/env python3
"""
needle_hunt.py — long-context needle retrieval test for vLLM OpenAI-compatible servers.

Written while chasing a deep-retrieval regression on GLM-5.2 (TP4/DCP4, MTP3,
nvfp4_ds_mla KV, 480k context). Two bugs in our earlier harnesses produced
false results for hours; both are fixed here:

  1. FALSE FAILURES. Some builds return the answer in the message's `reasoning`
     field with `content: null`. Harnesses that only read `content` (or that
     read `reasoning_content`, which this engine does not emit) report a MISS on
     a correct answer. This script searches `content`, `reasoning`,
     `reasoning_content`, `thinking`, and finally the whole serialized message.

  2. FALSE PASSES. Repeating the same prompt hits the server's prefix cache, so
     you re-measure a warm KV cache instead of real retrieval. We saw warm and
     cold disagree at 475k. Every depth here gets a unique prefix, and the
     script prints `cached=` from the server so you can verify it was cold.

It reports TWO independent verdicts, which are genuinely different failures:

  retrieval    — the needle value appears somewhere in the response
  finalization — a usable non-empty `content` field was produced

A model can retrieve correctly and still fail finalization (content is null),
which looks like an empty reply to any normal API client.

Usage
-----
  python3 needle_hunt.py --base http://localhost:8000 --model MyModel
  python3 needle_hunt.py --depths 50000,150000,300000,475000 --stop-on-miss
  python3 needle_hunt.py --save-json ./needle-out    # keep request/response JSON

Notes
-----
* Output is unbuffered and prints per depth, so you can stop early on failure
  instead of waiting for a long ladder to finish.
* `--effort` maps to chat_template_kwargs.reasoning_effort; omit with
  --no-effort if your model/template does not accept it.
* Needs only the standard library.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

# Neutral business filler. Repetitive on purpose: the needle must be found by
# retrieval, not by being the only interesting sentence in the window.
FILLER = (
    "The consolidation entries for the period were reviewed by the plant "
    "controllers before submission. Intercompany balances cleared without "
    "manual adjustment, and the hedging position rolled forward unchanged. "
    "Tooling amortization followed the agreed schedule, while indirect "
    "spend stayed within the corridor set at the last quarterly review. "
)
NEEDLE_VALUE = "738216"
NEEDLE = (
    "For reference, the maintenance ticket number for the Facility 27 "
    "compressor overhaul is 738216. "
)
QUESTION = (
    "\n\nFrom the document above: what is the maintenance ticket number for "
    "the Facility 27 compressor overhaul? Reply with the number only."
)


def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(base, model, text, timeout):
    """Prefer the server's tokenizer; fall back to a chars/4 estimate."""
    try:
        return post(f"{base}/tokenize", {"model": model, "prompt": text}, timeout)["count"]
    except Exception:
        return max(1, len(text) // 4)


def extract(msg):
    """Return (content, reasoning, found_anywhere) across known field names."""
    content = msg.get("content") or ""
    reasoning = (msg.get("reasoning")
                 or msg.get("reasoning_content")
                 or msg.get("thinking")
                 or "")
    anywhere = NEEDLE_VALUE in json.dumps(msg).replace(",", "")
    return content, reasoning, anywhere


def main():
    ap = argparse.ArgumentParser(description="Long-context needle retrieval test.")
    ap.add_argument("--base", default="http://localhost:8000",
                    help="server base URL (default: http://localhost:8000)")
    ap.add_argument("--model", default="GLM-5.2")
    ap.add_argument("--depths", default="50000,150000,250000,300000,350000,475000",
                    help="comma-separated approximate context sizes in tokens")
    ap.add_argument("--position", type=float, default=0.40,
                    help="needle depth as a fraction of the document (default 0.40). "
                         "Placement is DETERMINISTIC, not random.")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated positions to test at EVERY depth, "
                         "e.g. 0.1,0.4,0.8 . Failure mode can vary with position: "
                         "we observed clean refusal at 20%%, confabulation at 40%%, "
                         "and a degenerate repeat loop at 80%% — same context size.")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="generation budget; keep generous so reasoning cannot "
                         "starve the final answer (default 3000)")
    ap.add_argument("--effort", default="low",
                    help="chat_template_kwargs.reasoning_effort (default: low)")
    ap.add_argument("--no-effort", action="store_true",
                    help="omit chat_template_kwargs entirely")
    ap.add_argument("--runtag", default=str(int(time.time())),
                    help="unique tag mixed into each prompt to defeat prefix caching")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--stop-on-miss", action="store_true",
                    help="exit at the first retrieval miss")
    ap.add_argument("--save-json", default=None,
                    help="directory to write request/response JSON per depth")
    args = ap.parse_args()

    if args.save_json:
        os.makedirs(args.save_json, exist_ok=True)

    per = count_tokens(args.base, args.model, FILLER, args.timeout)
    print(f"# server={args.base} model={args.model} filler={per} tok/block "
          f"position={args.position:.0%} max_tokens={args.max_tokens} runtag={args.runtag}",
          flush=True)

    positions = ([float(x) for x in args.sweep.split(",")] if args.sweep
                 else [args.position])

    failures = 0
    cells = [(d, p) for d in (int(x) for x in args.depths.split(","))
             for p in positions]
    for depth, position in cells:
        blocks = max(4, depth // per)
        pre = int(blocks * position)
        # Unique prefix per depth+position+run: prevents the server's prefix
        # cache from serving a previous identical prompt and faking a pass.
        uniq = (f"RUNTAG {args.runtag} DEPTH {depth} POS {int(position*100)} "
                f"SEQ {depth * 7919 + int(position * 100)}.\n\n")
        doc = uniq + FILLER * pre + NEEDLE + FILLER * (blocks - pre)

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": doc + QUESTION}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        if not args.no_effort:
            payload["chat_template_kwargs"] = {"reasoning_effort": args.effort}

        tag = f"{depth}-pos{int(position*100)}"
        if args.save_json:
            with open(os.path.join(args.save_json, f"request-{tag}.json"), "w") as f:
                json.dump(payload, f)

        t0 = time.time()
        try:
            d = post(f"{args.base}/v1/chat/completions", payload, args.timeout)
        except Exception as e:
            print(f"depth={depth} pos={int(position*100)}% REQUEST_FAIL {e!r}", flush=True)
            failures += 1
            if args.stop_on_miss:
                sys.exit(2)
            continue
        secs = time.time() - t0

        if args.save_json:
            with open(os.path.join(args.save_json, f"response-{tag}.json"), "w") as f:
                json.dump(d, f, indent=2)

        ch = d["choices"][0]
        content, reasoning, anywhere = extract(ch["message"])
        in_content = NEEDLE_VALUE in content.replace(",", "")
        in_reasoning = NEEDLE_VALUE in reasoning.replace(",", "")
        where = ("content" if in_content else
                 "reasoning" if in_reasoning else
                 "other_field" if anywhere else "-")
        retrieval = "PASS" if anywhere else "MISS"
        finalization = "PASS" if content.strip() else "FAIL(content empty)"

        u = d.get("usage", {}) or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")

        print(f"depth={depth} pos={int(position*100)}% ctx={u.get('prompt_tokens')} cached={cached} "
              f"completion={u.get('completion_tokens')} finish={ch.get('finish_reason')} "
              f"secs={secs:.0f}", flush=True)
        print(f"   retrieval={retrieval} (where={where})   finalization={finalization}",
              flush=True)
        print(f"   content={content.strip()[:100]!r}", flush=True)
        if reasoning.strip():
            print(f"   reasoning_tail={reasoning.strip()[-110:]!r}", flush=True)

        if cached:
            print(f"   ! WARNING: {cached} prompt tokens served from prefix cache — "
                  f"not a cold measurement", flush=True)
        if ch.get("finish_reason") == "length":
            print("   ! WARNING: hit max_tokens; raise --max-tokens before trusting a MISS",
                  flush=True)

        if retrieval == "MISS":
            failures += 1
            if args.stop_on_miss:
                print("# stopping at first miss (--stop-on-miss)", flush=True)
                break

    print(f"# DONE failures={failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
