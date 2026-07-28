#!/usr/bin/env python3
"""Onset probe for finalized-content token duplication (v20 Gate 2 failure, 2026-07-25).

Gate 2 failed at its first rung: at 49,109 context tokens the model's `reasoning` quoted the
needle correctly (738216) but finalized `content` was `73838216` — a duplicated token — and the
same response duplicated a word in reasoning ("compressor compressor"). A 4,905-token request
answered exactly `738216`.

This probe answers three questions without restarting or reconfiguring anything:
  1. is it reproducible at a fixed depth (reps per cell)?
  2. where does it start (depth sweep)?
  3. is it confined to the answer, or does reasoning corrupt too?

Classification per response:
  EXACT      finalized content == needle
  DUP        needle recoverable but content has inserted/duplicated digits
  MISS       needle absent from content and reasoning
  REASON_OK  needle correct in reasoning, wrong in content  (the observed signature)
"""
import argparse, hashlib, json, os, pathlib, random, re, string, sys, time, urllib.request

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


def post(base, path, payload, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(base, text):
    return post(base, "/tokenize", {"model": MODEL, "prompt": text}, 300)["count"]


def head_text(mode, *, fixed_seed=None, salt=""):
    """none    : identical prefix across reps -> WARM (prefix cache hits), historical shape
       natural : unique but in-distribution -> COLD (cached_tokens 0)
       random  : unique 220 nonsense words -> COLD, but out of distribution (the 2026-07-25
                 instrument error; kept so its effect stays measurable rather than assumed)"""
    # Default behavior deliberately makes every natural/random request unique
    # for a cold-cache ladder. A fixed seed is an explicit deterministic replay
    # control: callers provide the same salt for all reps of one depth.
    rnd = (
        random.SystemRandom()
        if fixed_seed is None
        else random.Random(f"{fixed_seed}:{salt}")
    )
    if mode == "none":
        return ""
    if mode == "natural":
        pkt = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4)) + "-" + \
              "".join(rnd.choices(string.digits, k=6))
        return (f"Review packet {pkt} for the period ending 2026-{rnd.randint(1,12):02d}-"
                f"{rnd.randint(1,28):02d}, prepared by controller {rnd.randint(10,99)} of the "
                f"Facility {rnd.randint(2,26)} consolidation group. Packet sequence "
                f"{rnd.randint(1000,9999)}; supersedes revision {rnd.randint(1,9)}. ")
    return ("Audit reference tags for this review packet: " +
            " ".join("".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(4, 11)))
                     for _ in range(220)) + ". ")


def classify(content, reasoning, reasoning_content="", serialized_message=""):
    c = re.sub(r"[^0-9]", "", content or "")
    supporting_fields = "\n".join(
        value
        for value in (reasoning, reasoning_content, serialized_message)
        if isinstance(value, str)
    )
    r_has = NEEDLE in re.sub(r"[^0-9]", "", supporting_fields)
    if c == NEEDLE:
        return "EXACT"
    if NEEDLE in c or (len(c) > len(NEEDLE) and all(d in c for d in NEEDLE)):
        return "DUP"
    if r_has:
        return "REASON_OK"          # reasoning right, answer wrong: the observed signature
    return "MISS"


def word_dups(text):
    """Adjacent duplicated words, e.g. 'compressor compressor' — the reasoning-side signature."""
    w = re.findall(r"[a-zA-Z]+", text or "")
    return [w[i] for i in range(len(w) - 1) if w[i].lower() == w[i + 1].lower()]


SPEC_KEYS = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
             "spec_decode_num_accepted_tokens_total")


def spec_metrics(base):
    """Cumulative speculative-decode counters, so each request can report its own deltas
    (requested by Sol for the #171 extend-route hypothesis)."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=15) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for k in SPEC_KEYS:
            if line.startswith("vllm:" + k + "{"):
                out[k] = float(line.rsplit(" ", 1)[1])
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
            pos = line.split('position="')[1].split('"')[0]
            out[f"accepted_pos_{pos}"] = float(line.rsplit(" ", 1)[1])
    return out


def spec_delta(before, after):
    d = {k: round(after.get(k, 0) - before.get(k, 0), 2) for k in set(before) | set(after)}
    dt, acc = d.get("spec_decode_num_draft_tokens_total", 0), d.get("spec_decode_num_accepted_tokens_total", 0)
    d["acceptance_rate"] = round(acc / dt, 4) if dt else None
    return d


def probe(base, depth, per_block, rep, head_mode="natural", max_tokens=2000,
          fixed_head_seed=None, sampling_seed=None):
    head = head_text(
        head_mode,
        fixed_seed=fixed_head_seed,
        salt=f"{head_mode}:{depth}",
    )
    blocks = max(4, (depth - count_tokens(base, head + QUESTION + NEEDLE_SENT)) // per_block)
    pre = int(blocks * 0.40)
    prompt = head + FILLER * pre + NEEDLE_SENT + FILLER * (blocks - pre) + QUESTION
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"reasoning_effort": "low"}}
    if sampling_seed is not None:
        payload["seed"] = sampling_seed
    m_before = spec_metrics(base)
    t0 = time.time()
    resp = post(base, "/v1/chat/completions", payload, 3600)
    secs = round(time.time() - t0, 2)
    m_after = spec_metrics(base)
    ch = (resp.get("choices") or [{}])[0]; m = ch.get("message") or {}
    content = m.get("content") or ""
    reasoning = m.get("reasoning") or ""
    reasoning_content = m.get("reasoning_content") or ""
    serialized_message = json.dumps(m, sort_keys=True, ensure_ascii=False)
    combined_reasoning = reasoning or reasoning_content
    u = resp.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {"depth_target": depth, "rep": rep, "head_mode": head_mode,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "spec_delta": spec_delta(m_before, m_after),
            "context_tokens": u.get("prompt_tokens"),
            "cached_tokens": det.get("cached_tokens"), "completion_tokens": u.get("completion_tokens"),
            "finish_reason": ch.get("finish_reason"), "elapsed_s": secs,
            "verdict": classify(
                content,
                reasoning,
                reasoning_content,
                serialized_message,
            ),
            "content": content.strip(),
            "reasoning_word_dups": word_dups(combined_reasoning),
            "reasoning": reasoning,
            "reasoning_content": reasoning_content,
            "serialized_message": serialized_message,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--depths", type=int, nargs="+", default=[5_000, 15_000, 30_000, 50_000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/needle-onset"))
    ap.add_argument("--heads", nargs="+", default=["natural"],
                    choices=["none", "natural", "random"],
                    help="prompt-head modes to cross with depths (cold/warm, in/out of distribution)")
    ap.add_argument("--max-tokens", type=int, default=2000, dest="max_tokens")
    ap.add_argument(
        "--fixed-head-seed",
        type=int,
        help=(
            "reuse one deterministic head/prompt for every rep at a depth; "
            "default keeps cold requests unique"
        ),
    )
    ap.add_argument(
        "--sampling-seed",
        type=int,
        help="optional OpenAI request seed for exact replay controls",
    )
    a = ap.parse_args()
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    per_block = count_tokens(a.base, FILLER)
    rows = []
    for head_mode in a.heads:
      for depth in a.depths:
        for rep in range(1, a.reps + 1):
            try:
                r = probe(a.base, depth, per_block, rep, head_mode=head_mode,
                          max_tokens=a.max_tokens,
                          fixed_head_seed=a.fixed_head_seed,
                          sampling_seed=a.sampling_seed)
            except Exception as e:
                r = {"depth_target": depth, "rep": rep, "head_mode": head_mode,
                     "verdict": "ERROR", "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            print(f"[{head_mode:>7}] {depth//1000:>4}k rep{rep}: {r['verdict']:9} "
                  f"ctx={r.get('context_tokens')} fin={r.get('finish_reason')} "
                  f"cached={r.get('cached_tokens')} {r.get('elapsed_s')}s "
                  f"answer={r.get('content','')[:24]!r} "
                  f"dups={r.get('reasoning_word_dups', [])[:3]} "
                  f"accept={(r.get('spec_delta') or {}).get('acceptance_rate')} "
                  f"out_tok={r.get('completion_tokens')}", flush=True)
            (outdir / "rows.jsonl").open("a").write(json.dumps(r) + "\n")
    tally = {}
    for r in rows:
        tally.setdefault(f'{r.get("head_mode","?")}/{r["depth_target"]}', []).append(r["verdict"])
    print("\n=== tally ===", flush=True)
    for d, vs in tally.items():
        print(f"  {d:>16}: " + ", ".join(vs), flush=True)
    (outdir / "summary.json").write_text(json.dumps({"tally": {str(k): v for k, v in tally.items()},
                                                     "reps": a.reps}, indent=2) + "\n")


if __name__ == "__main__":
    main()
