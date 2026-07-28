#!/usr/bin/env python3
"""Is deep needle retrieval CONSISTENT on the no-think chat path? (Derek's 350k+ goal)

Established at 250k on NF3 5517197 (proofs#143):
  - The model retrieves 738216 correctly on every path tried.
  - With the default thinking template the model's emission of '</think>' (token 154842) is
    NONDETERMINISTIC for identical input at temperature 0, and the glm47_moe parser starts in
    REASONING with no fallback, so an omitted tag yields content=None.
  - With enable_thinking=false the prompt already carries the closed pair '<think></think>', and
    three cells across two endpoints and cold/warm produced BYTE-IDENTICAL output ids 3deb38ee3d70.

Single cell proves reachability, not consistency. This asks the question Derek actually posed:
does the chat API return the exact needle REPEATEDLY at 350k and 450k?

Each rep uses a DIFFERENT seed, so the document is different and the prefix cache cannot carry the
answer between reps. cached_tokens is recorded per rep and any rep with cached_tokens > 0 is
flagged, so a warm rep can never be mistaken for a cold pass.

Control arm (--with-thinking) runs the default thinking path at the same depths and seeds, so the
comparison is matched rather than against yesterday's numbers.
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


def count_tokens(base, text):
    return post(base, "/tokenize", {"model": MODEL, "prompt": text}, 300)["count"]


def build_prompt(base, depth, seed, per):
    rnd = random.Random(seed)
    pkt = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4)) + "-" + \
          "".join(rnd.choices(string.digits, k=6))
    head = (f"Review packet {pkt} for the period ending 2026-07-19, prepared by controller "
            f"{rnd.randint(10,99)} of the Facility {rnd.randint(2,26)} consolidation group. ")
    blocks = max(4, (depth - count_tokens(base, head + QUESTION + NEEDLE_SENT)) // per)
    pre = int(blocks * 0.40)
    return head + FILLER * pre + NEEDLE_SENT + FILLER * (blocks - pre) + QUESTION


def run_cell(base, prompt, thinking, max_tokens):
    kw = {} if thinking else {"enable_thinking": False}
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    if kw:
        payload["chat_template_kwargs"] = kw
    t0 = time.time()
    resp = post(base, "/v1/chat/completions", payload)
    return resp, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--depths", default="350000,450000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=760000)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--with-thinking", action="store_true",
                    help="also run the default thinking path at the same depths/seeds")
    ap.add_argument("--out", default="/tmp/nothink-consistency")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    per = count_tokens(a.base, FILLER)
    arms = [("nothink", False)] + ([("thinking", True)] if a.with_thinking else [])

    rows, tally = [], {}
    for depth in depths:
        for rep in range(1, a.reps + 1):
            seed = a.seed_base + depth + rep          # distinct document per (depth, rep)
            prompt = build_prompt(a.base, depth, seed, per)
            ctx = count_tokens(a.base, prompt)
            for arm, thinking in arms:
                try:
                    resp, secs = run_cell(a.base, prompt, thinking, a.max_tokens)
                except Exception as e:
                    print(f"{depth//1000}k rep{rep} {arm:8} ERROR {type(e).__name__}: {e}")
                    rows.append({"depth": depth, "rep": rep, "arm": arm, "verdict": "ERROR",
                                 "error": str(e)})
                    tally.setdefault((depth, arm), []).append("ERROR")
                    continue
                (outdir / f"resp-{depth}-r{rep}-{arm}.json").write_text(json.dumps(resp, indent=2))
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
                warm = "" if not cached else f" WARM(cached={cached})"
                print(f"{depth//1000}k rep{rep} {arm:8} {verdict:15} finish={ch.get('finish_reason')} "
                      f"out={u.get('completion_tokens')} ctx={ctx} "
                      f"content={content.strip()[:40]!r}{warm} {secs:.0f}s")
                rows.append({"depth": depth, "rep": rep, "arm": arm, "seed": seed, "ctx": ctx,
                             "verdict": verdict, "finish_reason": ch.get("finish_reason"),
                             "cached_tokens": cached, "usage": u, "content": content,
                             "reasoning_chars": len(reasoning), "secs": round(secs, 1)})
                tally.setdefault((depth, arm), []).append(verdict)
            (outdir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    print("\n===== CONSISTENCY SUMMARY =====")
    for (depth, arm), vs in sorted(tally.items()):
        ok = sum(1 for v in vs if v == "EXACT")
        print(f"  {depth//1000}k {arm:8} {ok}/{len(vs)} EXACT   {vs}")
    print("NOTHINK_CONSISTENCY_DONE")


if __name__ == "__main__":
    main()
