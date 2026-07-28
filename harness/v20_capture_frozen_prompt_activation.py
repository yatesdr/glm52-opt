#!/usr/bin/env python3
"""Send one hash-verified frozen prompt for activation capture.

This is deliberately not a quality gate.  It preserves the prompt and chat
template inputs from the frozen causal bundle while limiting decode to one
token, because the diagnostic artifact is produced during prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:5001")
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--expect-content")
    parser.add_argument("--expect-finish-reason")
    args = parser.parse_args()

    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    manifest = json.loads((args.freeze_dir / "manifest.json").read_text())
    matches = [
        row for row in manifest["prompts"] if row["label"] == args.label
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one manifest row for {args.label!r}")
    row = matches[0]
    prompt_path = args.freeze_dir / f"prompt-{args.label}.txt"
    prompt = prompt_path.read_text()
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    if prompt_sha256 != row["prompt_sha256"]:
        raise SystemExit(
            f"prompt SHA-256 mismatch: {prompt_sha256} != {row['prompt_sha256']}"
        )

    if not args.trace_dir.is_dir():
        raise SystemExit(f"trace directory does not exist: {args.trace_dir}")
    existing_trace_files = [
        path for path in args.trace_dir.rglob("*") if path.is_file()
    ]
    if existing_trace_files:
        raise SystemExit(
            "trace directory is not empty: "
            + ", ".join(str(path) for path in existing_trace_files[:8])
        )
    arm_path = args.trace_dir / "ARM"
    arm_path.touch(exist_ok=False)

    payload = {
        "model": "GLM-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": row["sampling"]["temperature"],
        "max_tokens": args.max_tokens,
    }
    if row["chat_template_kwargs"]:
        payload["chat_template_kwargs"] = row["chat_template_kwargs"]
    request = urllib.request.Request(
        args.base + row["endpoint"],
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.loads(response.read())
    finally:
        arm_path.unlink(missing_ok=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    usage = result.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    choice = (result.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    content = message.get("content")
    print(
        json.dumps(
            {
                "label": args.label,
                "prompt_sha256": prompt_sha256,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": cached,
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": finish_reason,
                "content": content,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if cached != 0:
        raise SystemExit("activation capture was not cold")
    if (
        args.expect_finish_reason is not None
        and finish_reason != args.expect_finish_reason
    ):
        raise SystemExit(
            "finish reason mismatch: "
            f"{finish_reason!r} != {args.expect_finish_reason!r}"
        )
    if args.expect_content is not None and content != args.expect_content:
        raise SystemExit(
            f"content mismatch: {content!r} != {args.expect_content!r}"
        )


if __name__ == "__main__":
    main()
