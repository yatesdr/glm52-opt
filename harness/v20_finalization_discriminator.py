#!/usr/bin/env python3
"""250k FINALIZATION discriminator (Sol dev#128, Derek's 350k+ goal).

At 250k on NF3 the needle IS retrieved — it appears in `reasoning` — but finalized `content` is
empty with finish_reason=stop and only 18 completion tokens. Retrieval works; finalization does not.

This changes ONLY finalization policy. Same process, same prompt bytes (fixed seed), same weights,
wire, MTP and KV config. Variants:

  chat-low     chat/completions, reasoning_effort=low   — reproduces the observed failure
  chat-high    chat/completions, reasoning_effort=high  — NF3 preset's own default
  chat-roomy   chat/completions, low effort, 8k tokens  — tests "ran out of room to finalize"
  raw          v1/completions                           — BYPASSES chat template + reasoning parser
                                                          entirely. If the needle appears here, the
                                                          model is fine and the chat layer is
                                                          dropping it.

Every raw response is written verbatim so token boundaries can be inspected afterwards.
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


def build_prompt(base, depth, seed):
    rnd = random.Random(seed)                      # FIXED seed: identical bytes across variants
    pkt = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4)) + "-" + \
          "".join(rnd.choices(string.digits, k=6))
    head = (f"Review packet {pkt} for the period ending 2026-07-19, prepared by controller "
            f"{rnd.randint(10,99)} of the Facility {rnd.randint(2,26)} consolidation group. ")
    per = count_tokens(base, FILLER)
    blocks = max(4, (depth - count_tokens(base, head + QUESTION + NEEDLE_SENT)) // per)
    pre = int(blocks * 0.40)
    return head + FILLER * pre + NEEDLE_SENT + FILLER * (blocks - pre) + QUESTION


def summarize(name, resp, secs, outdir):
    (outdir / f"raw-{name}.json").write_text(json.dumps(resp, indent=2))
    ch = (resp.get("choices") or [{}])[0]
    if "message" in ch:
        m = ch["message"]
        content = m.get("content") or ""
        reasoning = (m.get("reasoning") or "") + (m.get("reasoning_content") or "")
    else:                                           # /v1/completions shape
        content = ch.get("text") or ""
        reasoning = ""
    u = resp.get("usage") or {}
    digits = "".join(c for c in content if c.isdigit())
    verdict = ("EXACT" if digits == NEEDLE else
               "IN_CONTENT" if NEEDLE in content.replace(",", "") else
               "REASONING_ONLY" if NEEDLE in reasoning.replace(",", "") else
               "ABSENT")
    print(f"{name:12} {verdict:15} finish={ch.get('finish_reason')} "
          f"out_tok={u.get('completion_tokens')} content_chars={len(content)} "
          f"reasoning_chars={len(reasoning)} {secs:.0f}s")
    print(f"             content={content.strip()[:70]!r}")
    return {"variant": name, "verdict": verdict, "finish_reason": ch.get("finish_reason"),
            "usage": u, "content": content, "reasoning": reasoning, "secs": round(secs, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--depth", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--out", default="/tmp/finalization")
    ap.add_argument("--only", default="", help="comma-separated subset of variant names")
    a = ap.parse_args()
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(a.base, a.depth, a.seed)
    ctx = count_tokens(a.base, prompt)
    sys.stdout.reconfigure(line_buffering=True)
    print(f"[finalization] depth={a.depth} ctx={ctx} seed={a.seed} — identical bytes for all variants\n")

    variants = [
        ("chat-low",   "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                "max_tokens": 2000, "temperature": 0,
                                                "chat_template_kwargs": {"reasoning_effort": "low"}}),
        ("chat-high",  "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                "max_tokens": 2000, "temperature": 0,
                                                "chat_template_kwargs": {"reasoning_effort": "high"}}),
        ("chat-roomy", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                "max_tokens": 8000, "temperature": 0,
                                                "chat_template_kwargs": {"reasoning_effort": "low"}}),
        ("raw",        "/v1/completions",      {"model": MODEL, "prompt": prompt,
                                                "max_tokens": 2000, "temperature": 0}),
    ]
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        variants = [v for v in variants if v[0] in keep]
    rows = []
    for name, path, payload in variants:
        t0 = time.time()
        try:
            resp = post(a.base, path, payload)
        except Exception as e:
            print(f"{name:12} ERROR {type(e).__name__}: {e}")
            rows.append({"variant": name, "verdict": "ERROR", "error": str(e)})
            continue
        rows.append(summarize(name, resp, time.time() - t0, outdir))
    (outdir / "summary.json").write_text(json.dumps({"ctx": ctx, "depth": a.depth,
                                                     "seed": a.seed, "rows": rows}, indent=2) + "\n")
    print("\nFINALIZATION_PROBE_DONE")


if __name__ == "__main__":
    main()
