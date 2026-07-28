#!/usr/bin/env python3
"""Does `enable_thinking:false` recover finalized content at a depth where chat is broken?

Source-located defect (Fable, dev#135), read from the LIVE image and checkpoint:

  vllm/parser/glm47_moe.py:125
      initial_state = ParserState.REASONING if thinking else ParserState.CONTENT
  The only REASONING -> CONTENT transition is the literal terminal '</think>'. Generation that
  ends without emitting that token leaves message.content = None permanently.

  /model/chat_template.jinja:118
      <|assistant|>{{- '<think></think>' if (enable_thinking is defined and not enable_thinking)
                       else '<think>' -}}
  So generation ALWAYS starts inside an unclosed think block unless enable_thinking is false.

  glm47_moe.py:184-192 derives thinking_enabled from ONLY 'thinking'/'enable_thinking'.
  It cannot see 'reasoning_effort', which is the key our server actually sets.

Prediction under test: enable_thinking=false flips BOTH halves at once — the template emits a
CLOSED '<think></think>' and the parser starts in CONTENT — so content should populate. If it
does, the deep-context chat failure has a configuration fix and needs no patched parser.

This changes ONLY chat_template_kwargs. Same process, same weights, same wire, same MTP, same
prompt text, same fixed seed. Every response body is written verbatim.
"""
import argparse, json, pathlib, random, string, time, urllib.request

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
    rnd = random.Random(seed)
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
    else:
        content, reasoning = ch.get("text") or "", ""
    u = resp.get("usage") or {}
    digits = "".join(c for c in content if c.isdigit())
    verdict = ("EXACT" if digits == NEEDLE else
               "IN_CONTENT" if NEEDLE in content.replace(",", "") else
               "REASONING_ONLY" if NEEDLE in reasoning.replace(",", "") else
               "ABSENT")
    # The whole hypothesis turns on whether the closing terminal was ever emitted.
    closed = "</think>" in (content + reasoning)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"{name:16} {verdict:15} finish={ch.get('finish_reason')} "
          f"out_tok={u.get('completion_tokens')} content_chars={len(content)} "
          f"reasoning_chars={len(reasoning)} think_closed={closed} cached={cached} {secs:.0f}s",
          flush=True)
    print(f"                 content={content.strip()[:80]!r}", flush=True)
    return {"variant": name, "verdict": verdict, "finish_reason": ch.get("finish_reason"),
            "think_closed": closed, "cached_tokens": cached, "usage": u,
            "content": content, "reasoning": reasoning, "secs": round(secs, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--depth", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--out", default="/tmp/thinking-mode")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(a.base, a.depth, a.seed)
    ctx = count_tokens(a.base, prompt)
    print(f"[thinking-mode] depth={a.depth} ctx={ctx} seed={a.seed} max_tokens={a.max_tokens}\n",
          flush=True)

    msgs = [{"role": "user", "content": prompt}]
    base_chat = {"model": MODEL, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0}

    variants = [
        # Control: server default (reasoning_effort=high). Expected to fail as it has all evening.
        ("chat-default",  "/v1/chat/completions", {**base_chat}),
        # THE TEST: flips template:118 to a closed <think></think> AND parser to initial CONTENT.
        ("chat-nothink",  "/v1/chat/completions",
         {**base_chat, "chat_template_kwargs": {"enable_thinking": False}}),
        # Alternate key the parser also accepts; the template checks only enable_thinking, so this
        # separates "parser fixed" from "template fixed" if the two disagree.
        ("chat-thinkfls", "/v1/chat/completions",
         {**base_chat, "chat_template_kwargs": {"thinking": False}}),
        # Known-good floor for this depth.
        ("raw",           "/v1/completions",
         {"model": MODEL, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": 0}),
    ]
    keep = {s.strip() for s in a.only.split(",") if s.strip()}
    if keep:
        variants = [v for v in variants if v[0] in keep]

    rows = []
    for name, path, payload in variants:
        t0 = time.time()
        try:
            resp = post(a.base, path, payload)
        except Exception as e:
            print(f"{name:16} ERROR {type(e).__name__}: {e}", flush=True)
            rows.append({"variant": name, "verdict": "ERROR", "error": str(e)})
            continue
        rows.append(summarize(name, resp, time.time() - t0, outdir))
    (outdir / "summary.json").write_text(json.dumps(
        {"ctx": ctx, "depth": a.depth, "seed": a.seed, "rows": rows}, indent=2) + "\n")
    print("\nTHINKING_MODE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
