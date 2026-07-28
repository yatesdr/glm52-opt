#!/usr/bin/env python3
"""Separate GLM chat-template generation from response parsing.

The ordinary comparison

    /v1/chat/completions(messages=[...])  vs  /v1/completions(prompt="...")

does *not* give the model identical inputs: the chat endpoint renders the
checkpoint's chat template first.  This probe closes that gap:

1. render the chat request with ``/tokenize`` and retain the exact token IDs;
2. submit those IDs directly to ``/v1/completions``;
3. submit the normal chat request with the same template kwargs;
4. also run an explicit ``enable_thinking=false`` chat control.

The first two generated streams are model-input-identical.  If their token IDs
match but chat ``content`` is empty, response parsing/finalization is causal.
If they differ, the remaining serving-path difference (rather than the
template) is causal.  If both differ from the plain raw prompt, the template is
causal.

This is a diagnostic, not an acceptance harness.  It writes every response and
the rendered prompt IDs so parser behavior can be replayed off-GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import string
import sys
import time
import urllib.request
from typing import Any

NEEDLE = "738216"
NEEDLE_SENT = (
    "For reference, the maintenance ticket number for the Facility 27 "
    f"compressor overhaul is {NEEDLE}. "
)
FILLER = (
    "The consolidation entries for the period were reviewed by the plant "
    "controllers before submission. Intercompany balances cleared without "
    "manual adjustment, and the hedging position rolled forward unchanged. "
    "Tooling amortization followed the agreed schedule, while indirect "
    "spend stayed within the corridor set at the last quarterly review. "
)
QUESTION = (
    "\n\nFrom the document above: what is the maintenance ticket number for the "
    "Facility 27 compressor overhaul? Reply with the number only."
)
MODEL = "GLM-5.2"


def post(base: str, path: str, payload: dict[str, Any], timeout: int = 3600) -> dict:
    body = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def tokenize_completion(base: str, prompt: str) -> list[int]:
    response = post(
        base,
        "/tokenize",
        {"model": MODEL, "prompt": prompt, "add_special_tokens": True},
        300,
    )
    return response["tokens"]


def tokenize_chat(
    base: str, prompt: str, chat_template_kwargs: dict[str, Any]
) -> list[int]:
    response = post(
        base,
        "/tokenize",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "return_token_strs": False,
            "chat_template_kwargs": chat_template_kwargs,
        },
        300,
    )
    return response["tokens"]


def build_prompt(base: str, depth: int, seed: int) -> str:
    rnd = random.Random(seed)
    packet = "".join(rnd.choices(string.ascii_uppercase + string.digits, k=4))
    packet += "-"
    packet += "".join(rnd.choices(string.digits, k=6))
    head = (
        f"Review packet {packet} for the period ending 2026-07-19, prepared by "
        f"controller {rnd.randint(10, 99)} of the Facility "
        f"{rnd.randint(2, 26)} consolidation group. "
    )
    filler_tokens = len(tokenize_completion(base, FILLER))
    fixed_tokens = len(tokenize_completion(base, head + QUESTION + NEEDLE_SENT))
    blocks = max(4, (depth - fixed_tokens) // filler_tokens)
    before = int(blocks * 0.40)
    return head + FILLER * before + NEEDLE_SENT + FILLER * (blocks - before) + QUESTION


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def extract(row_name: str, response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message")
    if message is None:
        content = choice.get("text") or ""
        reasoning = ""
    else:
        content = message.get("content") or ""
        reasoning = (message.get("reasoning") or "") + (
            message.get("reasoning_content") or ""
        )
    token_ids = choice.get("token_ids") or []
    digits = "".join(char for char in content if char.isdigit())
    verdict = (
        "EXACT"
        if digits == NEEDLE
        else "IN_CONTENT"
        if NEEDLE in content.replace(",", "")
        else "REASONING_ONLY"
        if NEEDLE in reasoning.replace(",", "")
        else "ABSENT"
    )
    usage = response.get("usage") or {}
    return {
        "variant": row_name,
        "verdict": verdict,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "content": content,
        "reasoning": reasoning,
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:5001")
    parser.add_argument("--depth", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--out", default="/tmp/v20-chat-token-identity")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args.base, args.depth, args.seed)
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    thinking_kwargs = {"reasoning_effort": "high", "enable_thinking": True}
    no_thinking_kwargs = {"reasoning_effort": "none", "enable_thinking": False}
    plain_ids = tokenize_completion(args.base, prompt)
    chat_ids = tokenize_chat(args.base, prompt, thinking_kwargs)
    no_think_ids = tokenize_chat(args.base, prompt, no_thinking_kwargs)

    metadata = {
        "depth": args.depth,
        "seed": args.seed,
        "prompt_sha256": prompt_sha256,
        "plain_prompt_tokens": len(plain_ids),
        "chat_prompt_tokens": len(chat_ids),
        "no_think_prompt_tokens": len(no_think_ids),
        "plain_prompt_ids_sha256": sha256_json(plain_ids),
        "chat_prompt_ids_sha256": sha256_json(chat_ids),
        "no_think_prompt_ids_sha256": sha256_json(no_think_ids),
        "chat_suffix_ids": chat_ids[-32:],
        "no_think_suffix_ids": no_think_ids[-32:],
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (outdir / "chat-prompt-token-ids.json").write_text(json.dumps(chat_ids) + "\n")
    (outdir / "no-think-prompt-token-ids.json").write_text(
        json.dumps(no_think_ids) + "\n"
    )
    print(
        f"[identity] depth={args.depth} plain={len(plain_ids)} "
        f"chat={len(chat_ids)} no_think={len(no_think_ids)} "
        f"prompt_sha256={prompt_sha256}"
    )

    common = {
        "model": MODEL,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "return_token_ids": True,
        "skip_special_tokens": False,
    }
    variants: list[tuple[str, str, dict[str, Any]]] = [
        (
            "raw-plain",
            "/v1/completions",
            {**common, "prompt": plain_ids},
        ),
        (
            "raw-chat-ids",
            "/v1/completions",
            {**common, "prompt": chat_ids},
        ),
        (
            "chat-thinking",
            "/v1/chat/completions",
            {
                **common,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": thinking_kwargs,
            },
        ),
        (
            "raw-no-think-ids",
            "/v1/completions",
            {**common, "prompt": no_think_ids},
        ),
        (
            "chat-no-thinking",
            "/v1/chat/completions",
            {
                **common,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": no_thinking_kwargs,
            },
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, path, payload in variants:
        start = time.time()
        response = post(args.base, path, payload)
        (outdir / f"response-{name}.json").write_text(
            json.dumps(response, indent=2) + "\n"
        )
        row = extract(name, response)
        row["seconds"] = round(time.time() - start, 2)
        rows.append(row)
        print(
            f"{name:18} {row['verdict']:15} "
            f"finish={row['finish_reason']} out={row['completion_tokens']} "
            f"ids={len(row['token_ids'])} idsha={row['token_ids_sha256'][:12]} "
            f"{row['seconds']:.1f}s"
        )

    by_name = {row["variant"]: row for row in rows}
    raw_chat = by_name["raw-chat-ids"]["token_ids"]
    chat = by_name["chat-thinking"]["token_ids"]
    raw_no_think = by_name["raw-no-think-ids"]["token_ids"]
    chat_no_think = by_name["chat-no-thinking"]["token_ids"]
    comparison = {
        "raw_chat_ids_equal_chat_ids": bool(raw_chat) and raw_chat == chat,
        "raw_no_think_ids_equal_chat_no_think_ids": bool(raw_no_think)
        and raw_no_think == chat_no_think,
    }
    result = {"metadata": metadata, "rows": rows, "comparison": comparison}
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(comparison, sort_keys=True))
    print("CHAT_TOKEN_IDENTITY_PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
