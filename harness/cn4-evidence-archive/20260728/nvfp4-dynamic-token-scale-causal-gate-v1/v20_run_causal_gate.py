#!/usr/bin/env python3
"""Replay the frozen causal-gate prompt set (Sol dev#148/#151).

Reads the manifest produced by `v20_freeze_causal_gate_prompts.py` and replays each frozen prompt
verbatim from disk. **Fails closed** if a prompt's sha256 no longer matches its manifest entry — the
whole point of the gate is that the input is provably the same one stock failed on.

Gate rule per dev#148, applied per prompt:

    cold (cached_tokens == 0)  AND  finish_reason == "stop"
    AND  content.strip() == "738216"   exactly

An answer present only in `reasoning` is a FAIL, not a pass — that is the finalization defect from
`proofs#143` and it does not satisfy retrieval.

Verdict:
  CONFIRMED  every FAIL-role prompt recovered to EXACT, and the PASS control still passes.
  REFUTED    any FAIL-role prompt did not recover.
Control regression (control stops passing) invalidates the run rather than proving anything.

Order: PASS control first. It is the cheapest cell (250k vs 350k) so a fundamentally broken boot is
caught in ~5 minutes instead of ~25. The four frozen prompts differ at character 0 (random header),
so they share no prefix and cannot warm each other's cache; `cached_tokens` is still checked per
cell rather than assumed.
"""
import argparse, hashlib, json, pathlib, sys, time, urllib.request

NEEDLE = "738216"
MODEL = "GLM-5.2"


def post(base, path, payload, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5001")
    ap.add_argument("--freeze-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default="(unspecified)")
    ap.add_argument("--note", default="")
    ap.add_argument(
        "--labels",
        default="",
        help=(
            "comma-separated manifest labels to run after preserving manifest "
            "order; omitted runs the complete frozen gate"
        ),
    )
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    fdir = pathlib.Path(a.freeze_dir)
    manifest = json.loads((fdir / "manifest.json").read_text())
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[gate] image={a.image}")
    print(f"[gate] freeze={fdir}  stock_image={manifest.get('stock_image')}")
    if a.note:
        print(f"[gate] note={a.note}")
    print()

    entries = manifest["prompts"]
    order = [e for e in entries if e["role"] == "PASS_CONTROL"] + \
            [e for e in entries if e["role"] == "FAIL"]
    if a.labels:
        requested = [label.strip() for label in a.labels.split(",") if label.strip()]
        known = {e["label"] for e in order}
        unknown = [label for label in requested if label not in known]
        if unknown:
            raise SystemExit(f"unknown frozen prompt label(s): {', '.join(unknown)}")
        requested_set = set(requested)
        order = [e for e in order if e["label"] in requested_set]
        if len(order) != len(requested_set):
            raise SystemExit("frozen prompt selection is not one-to-one")

    rows = []
    for e in order:
        label = e["label"]
        prompt = (fdir / f"prompt-{label}.txt").read_text()
        actual = hashlib.sha256(prompt.encode()).hexdigest()
        if actual != e["prompt_sha256"]:
            print(f"{label:15} GATE_ABORT prompt sha256 mismatch\n"
                  f"{'':15}   manifest={e['prompt_sha256']}\n{'':15}   ondisk  ={actual}")
            sys.exit(2)

        payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                   "temperature": e["sampling"]["temperature"],
                   "max_tokens": e["sampling"]["max_tokens"]}
        if e["chat_template_kwargs"]:
            payload["chat_template_kwargs"] = e["chat_template_kwargs"]

        t0 = time.time()
        try:
            resp = post(a.base, e["endpoint"], payload)
        except Exception as ex:
            print(f"{label:15} ERROR {type(ex).__name__}: {ex}")
            rows.append({**{k: e[k] for k in ("label", "role", "depth", "seed",
                                              "prompt_sha256", "stock_verdict")},
                         "verdict": "ERROR", "error": str(ex)})
            continue
        secs = time.time() - t0

        (outdir / f"resp-{label}.json").write_text(json.dumps(resp, indent=2))
        ch = (resp.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        content = m.get("content") or ""
        reasoning = (m.get("reasoning") or "") + (m.get("reasoning_content") or "")
        u = resp.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        finish = ch.get("finish_reason")

        exact = content.strip() == NEEDLE
        cold = (cached == 0)
        passed = exact and cold and finish == "stop"
        verdict = ("EXACT" if exact else
                   "IN_CONTENT" if NEEDLE in content.replace(",", "") else
                   "REASONING_ONLY" if NEEDLE in reasoning.replace(",", "") else
                   "ABSENT")

        print(f"{label:15} {verdict:15} gate={'PASS' if passed else 'FAIL':4} "
              f"finish={finish} out={u.get('completion_tokens')} "
              f"prompt_tok={u.get('prompt_tokens')} cached={cached} "
              f"stock={e['stock_verdict']} {secs:.0f}s")
        print(f"{'':15} content={content.strip()[:90]!r}")
        if not cold:
            print(f"{'':15} *** NOT COLD — cell does not satisfy the gate regardless of verdict ***")

        rows.append({**{k: e[k] for k in ("label", "role", "depth", "seed",
                                          "prompt_sha256", "stock_verdict")},
                     "verdict": verdict, "gate_pass": passed, "exact": exact, "cold": cold,
                     "finish_reason": finish, "cached_tokens": cached, "usage": u,
                     "content": content, "reasoning_chars": len(reasoning),
                     "secs": round(secs, 1)})
        (outdir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    fails = [r for r in rows if r["role"] == "FAIL"]
    ctl = [r for r in rows if r["role"] == "PASS_CONTROL"]
    recovered = [r for r in fails if r.get("gate_pass")]
    ctl_ok = all(r.get("gate_pass") for r in ctl) if ctl else None

    print("\n===== CAUSAL GATE SUMMARY =====")
    print(f"  FAIL-role prompts recovered: {len(recovered)}/{len(fails)}")
    print(f"  PASS control still passing:   {ctl_ok}")
    if ctl_ok is False:
        verdict = "INVALID — PASS control regressed; run proves nothing about the FAIL set"
    elif fails and len(recovered) == len(fails):
        verdict = "CONFIRMED — every frozen stock-FAIL prompt recovered"
    else:
        verdict = "REFUTED — at least one frozen stock-FAIL prompt did not recover"
    print(f"  VERDICT: {verdict}")

    (outdir / "summary.json").write_text(json.dumps({
        "image": a.image, "note": a.note, "freeze_dir": str(fdir),
        "stock_image": manifest.get("stock_image"),
        "recovered": len(recovered), "fail_count": len(fails),
        "control_ok": ctl_ok, "verdict": verdict, "rows": rows,
    }, indent=2) + "\n")
    print("CAUSAL_GATE_DONE")


if __name__ == "__main__":
    main()
