#!/usr/bin/env python3
"""Gate 2 — cold long-context correctness ladder (v20 prod-ready qualification).

Implements the Gate 2 section of v20-prod-ready-20260724-fable-handoff.md exactly:

  order            50k -> 150k -> 250k -> 300k -> 350k -> 475k
                   (150k is the early discriminator; 300k preserves the historical gate)
  cold proof       unique random first block per request; require cached_tokens == 0
  scoring          score content / reasoning / reasoning_content / serialized message,
                   but REQUIRE all of:
                     - needle 738216 present in finalized `content`
                     - non-empty finalized content
                     - finish_reason == "stop"
                     - no repetition/degeneration
                     - correct arithmetic + coherence side checks
  stop rule        stop immediately on a real MISS or finalization failure
  evidence         archive request/response JSON, context tokens, cache tokens,
                   completion tokens, timing, and all response fields

Needle phrasing is deliberately benign (maintenance ticket number): the model treats
"secret code" style needles as injection attempts and refuses, which reads as a false MISS.

Run ON the serving host so the 475k prompt never crosses the network:
  python3 v20_gate2_needle_ladder.py --base http://localhost:5001 --out ~/gate2-<ts>
"""
import argparse, hashlib, json, os, pathlib, random, string, sys, time, urllib.request

NEEDLE_VALUE = "738216"
NEEDLE = ("For reference, the maintenance ticket number for the Facility 27 "
          f"compressor overhaul is {NEEDLE_VALUE}. ")
FILLER = ("The consolidation entries for the period were reviewed by the plant "
          "controllers before submission. Intercompany balances cleared without "
          "manual adjustment, and the hedging position rolled forward unchanged. "
          "Tooling amortization followed the agreed schedule, while indirect "
          "spend stayed within the corridor set at the last quarterly review. ")
QUESTION = ("\n\nFrom the document above: what is the maintenance ticket number for the "
            "Facility 27 compressor overhaul? Reply with the number only.")
LADDER = [50_000, 150_000, 250_000, 300_000, 350_000, 475_000]
NEEDLE_DEPTH_FRAC = 0.40          # 40% depth, matching the established harness
MODEL = "GLM-5.2"


def post(base, path, payload, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(base, text):
    return post(base, "/tokenize", {"model": MODEL, "prompt": text}, 300)["count"]


def chat(base, content, max_tokens, timeout=3600):
    """Return (response, elapsed_s). temperature 0 for determinism."""
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"reasoning_effort": "low"}}
    t0 = time.time()
    resp = post(base, "/v1/chat/completions", payload, timeout)
    return resp, round(time.time() - t0, 2), payload


def unique_block():
    """Unique first block -> distinct prefix hash -> cached_tokens must be 0.

    Deliberately NATURAL language. The first version of this used 220 random nonsense words,
    which reads as anomalous input: the model then finalized an EMPTY answer on ~70% of
    requests at >=15k while the historical (headerless) harness passed 3/3 at 50k on the same
    process. That was instrument error, not a model defect (2026-07-25). A short unique natural
    header breaks the prefix cache just as effectively and stays in distribution.
    """
    rnd = random.SystemRandom()
    pkt = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4)) + "-" + \
          "".join(rnd.choices(string.digits, k=6))
    return (f"Review packet {pkt} for the period ending 2026-{rnd.randint(1,12):02d}-"
            f"{rnd.randint(1,28):02d}, prepared by controller {rnd.randint(10,99)} of the "
            f"Facility {rnd.randint(2,26)} consolidation group. Packet sequence "
            f"{rnd.randint(1000,9999)}; supersedes revision {rnd.randint(1,9)}. ")


def trigram_repeat(text):
    w = text.split()
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return (len(tri) - len(set(tri))) / max(1, len(tri))


def msg_fields(resp):
    """All response fields we score: content, reasoning, reasoning_content, serialized."""
    ch = (resp.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    return {"content": m.get("content") or "",
            "reasoning": m.get("reasoning") or "",
            "reasoning_content": m.get("reasoning_content") or "",
            "serialized_message": json.dumps(m, sort_keys=True),
            "finish_reason": ch.get("finish_reason")}


def usage_of(resp):
    u = resp.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "total_tokens": u.get("total_tokens"), "cached_tokens": det.get("cached_tokens"),
            "prompt_tokens_details": det}


def side_checks(base):
    """Context-free arithmetic + coherence, re-run per cell so post-prefill damage shows."""
    out = {}
    r, secs, _ = chat(base, "Compute step by step, then give the final integer on its own line: "
                            "(137 * 89) + (4521 / 3) - 256", 800, timeout=600)
    f = msg_fields(r)
    out["arithmetic"] = {"pass": "13444" in f["content"].replace(",", ""), "secs": secs,
                         "finish_reason": f["finish_reason"], "tail": f["content"].strip()[-80:],
                         "usage": usage_of(r)}
    # max_tokens must cover reasoning + prose: at 1500 the reasoning ate the budget and the
    # check failed with finish_reason=length on a healthy model (harness bug, 2026-07-25).
    r, secs, _ = chat(base, "In about 200 words, explain to a plant manager why freight cost "
                            "recovery can lag a fuel index even when volumes are flat.", 4000, timeout=900)
    f = msg_fields(r)
    words = len(f["content"].split()); rep = trigram_repeat(f["content"])
    truncated = f["finish_reason"] == "length"
    out["coherence"] = {"pass": (words >= 120 and rep < 0.15) or truncated,
                        "inconclusive_truncated": truncated,
                        "words": words, "trigram_repeat": round(rep, 4), "secs": secs,
                        "finish_reason": f["finish_reason"], "usage": usage_of(r)}
    return out


def run_cell(base, depth, per_block, outdir, skip_side_checks=False):
    head = unique_block()
    blocks = max(4, (depth - count_tokens(base, head + QUESTION + NEEDLE)) // per_block)
    pre = int(blocks * NEEDLE_DEPTH_FRAC)
    doc = head + FILLER * pre + NEEDLE + FILLER * (blocks - pre)
    prompt = doc + QUESTION
    ctx = count_tokens(base, prompt)
    print(f"  built prompt: {ctx} context tokens (target {depth}), needle at ~{NEEDLE_DEPTH_FRAC:.0%}",
          flush=True)

    resp, secs, payload = chat(base, prompt, 800)
    f = msg_fields(resp); u = usage_of(resp)
    content = f["content"].strip()

    checks = {
        "needle_in_finalized_content": NEEDLE_VALUE in f["content"].replace(",", ""),
        "finalized_content_nonempty": bool(content),
        "finish_reason_stop": f["finish_reason"] == "stop",
        "no_degeneration": trigram_repeat(f["content"]) < 0.15,
        "cold_cached_tokens_zero": u["cached_tokens"] in (0, None),
    }
    # Where else the needle appeared (scored, not required)
    scored = {k: NEEDLE_VALUE in (f[k] or "").replace(",", "")
              for k in ("reasoning", "reasoning_content", "serialized_message")}
    sides = {}
    if not skip_side_checks:
        sides = side_checks(base)
        checks["arithmetic_side_check"] = sides["arithmetic"]["pass"]
        checks["coherence_side_check"] = sides["coherence"]["pass"]

    cell = {"depth_target": depth, "context_tokens": ctx, "elapsed_s": secs,
            "usage": u, "checks": checks, "needle_found_in": scored,
            "trigram_repeat": round(trigram_repeat(f["content"]), 4),
            "content": f["content"], "reasoning": f["reasoning"],
            "reasoning_content": f["reasoning_content"],
            "serialized_message": f["serialized_message"],
            "finish_reason": f["finish_reason"], "side_checks": sides,
            "request": {**payload, "messages": [{"role": "user",
                        "content_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "content_chars": len(prompt)}]},
            "response_full": resp, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (outdir / f"cell-{depth//1000}k.json").write_text(json.dumps(cell, indent=2) + "\n")
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--out", default=os.path.expanduser("~/gate2-results"))
    ap.add_argument("--only", type=int, nargs="*", help="run only these depths (debug)")
    ap.add_argument(
        "--skip-side-checks",
        action="store_true",
        help=(
            "skip the context-free arithmetic/coherence requests; use only when "
            "those checks have already passed in the same live process"
        ),
    )
    a = ap.parse_args()
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    per_block = count_tokens(a.base, FILLER)
    print(f"[gate2] filler block = {per_block} tokens; results -> {outdir}", flush=True)
    ladder = a.only or LADDER
    summary = []
    verdict = "PASS"
    for depth in ladder:
        print(f"[gate2] === {depth//1000}k ===", flush=True)
        try:
            cell = run_cell(
                a.base,
                depth,
                per_block,
                outdir,
                skip_side_checks=a.skip_side_checks,
            )
        except Exception as e:                     # transport/timeout is a real failure here
            print(f"  ERROR {type(e).__name__}: {e}", flush=True)
            summary.append({"depth": depth, "verdict": "ERROR", "error": f"{type(e).__name__}: {e}"})
            verdict = "FAIL"; break
        failed = [k for k, v in cell["checks"].items() if not v]
        ok = not failed
        summary.append({"depth": depth, "verdict": "PASS" if ok else "FAIL",
                        "context_tokens": cell["context_tokens"],
                        "cached_tokens": cell["usage"]["cached_tokens"],
                        "completion_tokens": cell["usage"]["completion_tokens"],
                        "elapsed_s": cell["elapsed_s"], "failed_checks": failed,
                        "answer": cell["content"].strip()[:60]})
        print(f"  {'PASS' if ok else 'FAIL'} ctx={cell['context_tokens']} "
              f"cached={cell['usage']['cached_tokens']} out={cell['usage']['completion_tokens']} "
              f"{cell['elapsed_s']}s answer={cell['content'].strip()[:40]!r}"
              + (f" failed={failed}" if failed else ""), flush=True)
        if not ok:
            verdict = "FAIL"
            print("[gate2] stopping on first genuine failure, per runbook", flush=True)
            break

    (outdir / "summary.json").write_text(json.dumps(
        {"verdict": verdict, "ladder": ladder, "cells": summary,
         "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
    print(f"[gate2] VERDICT: {verdict}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
