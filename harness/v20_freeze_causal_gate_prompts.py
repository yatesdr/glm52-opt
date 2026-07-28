#!/usr/bin/env python3
"""Freeze the causal-gate prompt set for the legacy-projection discriminator (Sol dev#148/#151).

Produces a single JSON manifest that pins, for each gate prompt:
  * the exact prompt text and its sha256
  * the CHAT-RENDERED input token ids (via /tokenize on the chat request) and their sha256
  * the exact sampling fields the gate must replay
  * the observed stock verdict, so the A/B is scored against a recorded baseline

FAIL set — three cold 350k no-think prompts, stock verdict ABSENT (0/3 in
`nothink-consistency-20260726T0125Z`, all cached_tokens=0). Seeds are the ladder's
`760000 + depth + rep`.

PASS control — 250k no-think, stock verdict EXACT with content exactly '738216', cold, 4 tokens,
generated-id sha `3deb38ee3d70` (`thinkmode-250k-20260726T0105Z/raw-chat-nothink.json`).

  Why 250k and not 450k: the 450k no-think cells returned IN_CONTENT — correct needle inside a full
  sentence. Under the gate's "exact non-empty content == 738216" rule those would fail on STOCK,
  making them useless as a non-regression control. The 250k cell returned the bare digits.

Prompt construction is byte-identical to `v20_nothink_consistency_ladder.py` /
`v20_thinking_mode_finalization_fix.py` — same FILLER, NEEDLE_SENT, QUESTION, head format, 40%
placement, and the same `per = ntok(FILLER)` block arithmetic. Verified by re-deriving ctx and
needle offset here and comparing against the recorded stock values.

Read-only against the server: /tokenize only. No generation, no config change.
"""
import argparse, hashlib, json, pathlib, random, string, sys, urllib.request

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

# (label, depth, seed, arm, stock_verdict, expected_ctx, note)
GATE = [
    ("fail-350k-r1", 350000, 760000 + 350000 + 1, "nothink", "ABSENT", 343721,
     "stock: content fabricated 'MNT-2024-087', 138 tok, finish=stop, cached=0"),
    ("fail-350k-r2", 350000, 760000 + 350000 + 2, "nothink", "ABSENT", 343721,
     "stock: 23 tok, finish=stop, cached=0"),
    ("fail-350k-r3", 350000, 760000 + 350000 + 3, "nothink", "ABSENT", 343721,
     "stock: 26 tok, finish=stop, cached=0"),
    ("pass-250k-ctl", 250000, 20260725, "nothink", "EXACT", 245491,
     "stock: content=='738216', 4 tok, finish=stop, cached=0, gen-ids sha 3deb38ee3d70"),
]

SAMPLING = {"temperature": 0, "max_tokens": 2000}


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(base, text):
    return post(base, "/tokenize", {"model": MODEL, "prompt": text})["count"]


def build_prompt(base, depth, seed, per):
    rnd = random.Random(seed)
    pkt = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4)) + "-" + \
          "".join(rnd.choices(string.digits, k=6))
    head = (f"Review packet {pkt} for the period ending 2026-07-19, prepared by controller "
            f"{rnd.randint(10,99)} of the Facility {rnd.randint(2,26)} consolidation group. ")
    blocks = max(4, (depth - ntok(base, head + QUESTION + NEEDLE_SENT)) // per)
    pre = int(blocks * 0.40)
    return head + FILLER * pre + NEEDLE_SENT + FILLER * (blocks - pre) + QUESTION


def chat_render_ids(base, prompt, arm):
    """Tokenize the chat request exactly as the server will render it."""
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    if arm == "nothink":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = post(base, "/tokenize", payload)
    ids = r.get("tokens")
    if ids is None:                     # fail closed rather than silently pin nothing
        raise RuntimeError(f"/tokenize returned no token ids for arm={arm}: keys={list(r)}")
    return ids, r.get("count")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--out", default="/tmp/causal-gate-freeze")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    per = ntok(a.base, FILLER)
    print(f"[freeze] filler_tokens={per}\n")

    manifest, mismatch = [], []
    for label, depth, seed, arm, stock, exp_ctx, note in GATE:
        prompt = build_prompt(a.base, depth, seed, per)
        ctx = ntok(a.base, prompt)
        psha = hashlib.sha256(prompt.encode()).hexdigest()
        ids, idcount = chat_render_ids(a.base, prompt, arm)
        isha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        idx = prompt.find(NEEDLE)
        tok_before = ntok(a.base, prompt[:idx]) if idx >= 0 else -1

        ok = (ctx == exp_ctx) and prompt.count(NEEDLE) == 1
        if not ok:
            mismatch.append((label, ctx, exp_ctx, prompt.count(NEEDLE)))
        print(f"{label:15} ctx={ctx} (expect {exp_ctx}) {'OK' if ok else 'MISMATCH'}  "
              f"needles={prompt.count(NEEDLE)} needle_at_tok={tok_before} "
              f"({100*tok_before/ctx:.1f}%)")
        print(f"{'':15} prompt_sha256={psha}")
        print(f"{'':15} rendered_ids={idcount} ids_sha256={isha}")
        print(f"{'':15} stock={stock} — {note}")

        (outdir / f"prompt-{label}.txt").write_text(prompt)
        (outdir / f"ids-{label}.json").write_text(json.dumps(ids))
        manifest.append({
            "label": label, "depth": depth, "seed": seed, "arm": arm,
            "role": "FAIL" if stock != "EXACT" else "PASS_CONTROL",
            "prompt_sha256": psha, "prompt_chars": len(prompt),
            "prompt_tokens": ctx, "expected_ctx": exp_ctx, "ctx_matches_stock": ok,
            "needle_count": prompt.count(NEEDLE), "needle_token_offset": tok_before,
            "rendered_input_ids_count": idcount, "rendered_input_ids_sha256": isha,
            "chat_template_kwargs": ({"enable_thinking": False} if arm == "nothink" else {}),
            "sampling": SAMPLING, "endpoint": "/v1/chat/completions",
            "stock_verdict": stock, "stock_note": note,
            "gate_requirement": "cold (cached_tokens=0), finish_reason=stop, "
                                "content == '738216' exactly; needle in reasoning only = FAIL",
        })

    (outdir / "manifest.json").write_text(json.dumps({
        "purpose": "legacy-projection causal gate (Sol dev#148/#151)",
        "stock_image": "glm52-serve:v20-5517197-pxb-20260725",
        "stock_image_id": "sha256:2566f905f13252c514a0f96c177ba982bd16321943927966310bf8c7c92d94b7",
        "filler_tokens": per, "needle": NEEDLE,
        "token_decode": {"738216": [22, 100919, 122250], "<think>": 154841,
                         "</think>": 154842, "eos": 154827},
        "prompts": manifest,
    }, indent=2) + "\n")

    if mismatch:
        print(f"\nFREEZE_FAILED — ctx/needle mismatch vs recorded stock: {mismatch}")
        sys.exit(1)
    print("\nFREEZE_OK — all prompts reproduce the recorded stock ctx and needle placement")


if __name__ == "__main__":
    main()
