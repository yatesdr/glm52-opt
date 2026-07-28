#!/usr/bin/env python3
"""Is deep-context failure REPRODUCIBLE per prompt, or nondeterministic run-to-run?

This is the question that decides whether the v20 long-context defect can be bisected.

Established 2026-07-26 on NF3 5517197, cold, ctx=343,721 for every prompt below:
  seed 20260725  -> EXACT (738216)
  seed 1110001   -> total failure (model: "the document content is not provided in the prompt")
  seed 1110002   -> total failure
Verified identical structure: needle present exactly once, at token 137,496, 40.0% depth, in all
three. Only ~90 characters of random header text at position 0 differ.

  If each prompt reproduces its own verdict  -> deterministic per prompt; bisectable.
  If verdicts flip between reps              -> genuine nondeterminism; bisection is much harder.

CACHE HANDLING. `--enable-prefix-caching` is on, so re-sending an identical prompt would score a
near-total prefix hit and would not be an independent trial. This probe therefore INTERLEAVES the
prompts (A,B,C,A,B,C,...) rather than repeating each in place. With a KV pool of 559,616 tokens and
each prompt at ~343,721, any two consecutive full-length prompts exceed the pool, so each return to
a given prompt is substantially evicted. `cached_tokens` is recorded and printed for every cell so
the actual coldness of each trial is visible rather than assumed -- never infer it.
"""
import argparse, json, pathlib, random, string, sys, time, urllib.request

NEEDLE = "738216"
NEEDLE_SENT = ("For reference, the maintenance ticket number for the Facility 27 "
               f"compressor overhaul is {NEEDLE}. ")
FILLER = ("The consolidation entries for the period were reviewed by the plant "
          "controllers before submission. Intercompany balances cleared without "
          "manual adjustment, and the hedging position rolled forward unchanged. "
          "Tooling amortization followed the agreed schedule, while indirect "
          "spend stayed within the corridor set at the last quarterly review. ")
QUESTION = ("\n\nFrom the document above: what is the maintenance ticket number for the "
            "Facility 27 compressor overhaul? Reply with the number only.")
MODEL = "GLM-5.2"


def post(base, path, payload, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(base, s):
    return post(base, "/tokenize", {"model": MODEL, "prompt": s}, 300)["count"]


def build_prompt(base, depth, seed, per):
    rnd = random.Random(seed)
    pkt = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4)) + "-" + \
          "".join(rnd.choices(string.digits, k=6))
    head = (f"Review packet {pkt} for the period ending 2026-07-19, prepared by controller "
            f"{rnd.randint(10,99)} of the Facility {rnd.randint(2,26)} consolidation group. ")
    blocks = max(4, (depth - ntok(base, head + QUESTION + NEEDLE_SENT)) // per)
    pre = int(blocks * 0.40)
    return head + FILLER * pre + NEEDLE_SENT + FILLER * (blocks - pre) + QUESTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--depth", type=int, default=350000)
    ap.add_argument("--seeds", default="20260725,1110001,1110002")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--thinking", action="store_true",
                    help="use the default thinking template (default: enable_thinking=false)")
    ap.add_argument("--out", default="/tmp/repro")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    per = ntok(a.base, FILLER)

    prompts = {}
    for s in seeds:
        p = build_prompt(a.base, a.depth, s, per)
        prompts[s] = p
        print(f"[build] seed={s} ctx={ntok(a.base, p)} needles={p.count(NEEDLE)}")
    print()

    rows, tally = [], {s: [] for s in seeds}
    for rep in range(1, a.reps + 1):
        for s in seeds:                                  # interleaved, not repeated in place
            payload = {"model": MODEL,
                       "messages": [{"role": "user", "content": prompts[s]}],
                       "max_tokens": a.max_tokens, "temperature": 0}
            if not a.thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            t0 = time.time()
            try:
                resp = post(a.base, "/v1/chat/completions", payload)
            except Exception as e:
                print(f"rep{rep} seed={s} ERROR {type(e).__name__}: {e}")
                tally[s].append("ERROR")
                continue
            secs = time.time() - t0
            (outdir / f"resp-s{s}-r{rep}.json").write_text(json.dumps(resp, indent=2))
            ch = (resp.get("choices") or [{}])[0]
            m = ch.get("message") or {}
            content = m.get("content") or ""
            reasoning = (m.get("reasoning") or "") + (m.get("reasoning_content") or "")
            u = resp.get("usage") or {}
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            digits = "".join(c for c in content if c.isdigit())
            verdict = ("EXACT" if digits == NEEDLE else
                       "IN_CONTENT" if NEEDLE in content.replace(",", "") else
                       "REASONING_ONLY" if NEEDLE in reasoning.replace(",", "") else
                       "ABSENT")
            tally[s].append(verdict)
            rows.append({"seed": s, "rep": rep, "verdict": verdict, "cached_tokens": cached,
                         "finish_reason": ch.get("finish_reason"), "usage": u,
                         "content": content, "reasoning_chars": len(reasoning),
                         "secs": round(secs, 1)})
            print(f"rep{rep} seed={s:<9} {verdict:15} finish={ch.get('finish_reason')} "
                  f"out={u.get('completion_tokens')} cached={cached} "
                  f"content={content.strip()[:40]!r} {secs:.0f}s")
            (outdir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    print("\n===== REPRODUCIBILITY SUMMARY =====")
    for s in seeds:
        vs = tally[s]
        stable = "STABLE" if len(set(vs)) == 1 else "FLIPS"
        print(f"  seed {s:<9} {stable:7} {vs}")
    print("REPRO_PROBE_DONE")


if __name__ == "__main__":
    main()
