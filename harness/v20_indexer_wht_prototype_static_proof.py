#!/usr/bin/env python3
"""Fail-closed source proof for the GLM indexer WHT causal prototype."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(tree: ast.AST, class_name: str | None, name: str) -> ast.FunctionDef:
    body: list[ast.stmt]
    if class_name is None:
        body = tree.body  # type: ignore[attr-defined]
    else:
        cls = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = cls.body
    return next(
        node
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(function: ast.FunctionDef) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            cursor: ast.expr = node.func
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            name = ".".join(reversed(parts))
        calls.append((name, node.lineno))
    return calls


def run(config_path: Path, model_path: Path) -> dict[str, Any]:
    config_tree = ast.parse(config_path.read_text(), filename=str(config_path))
    model_tree = ast.parse(model_path.read_text(), filename=str(model_path))

    annotation = next(
        node
        for node in config_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "IndexerFP8Conditioning"
            for target in node.targets
        )
    )
    literal_values = {
        node.value
        for node in ast.walk(annotation.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if literal_values != {"none", "hadamard"}:
        raise RuntimeError("conditioning Literal does not contain none/hadamard")

    attention_class = next(
        node
        for node in config_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AttentionConfig"
    )
    field_node = next(
        node
        for node in attention_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "indexer_fp8_conditioning"
    )
    if not isinstance(field_node.value, ast.Constant) or field_node.value.value != "none":
        raise RuntimeError("prototype must remain disabled by default")

    helper = _function(model_tree, None, "_condition_indexer_fp8_row")
    helper_calls = _calls(helper)
    hadacore = [line for name, line in helper_calls if name == "ops.hadacore_transform"]
    if len(hadacore) != 1:
        raise RuntimeError(f"expected one Hadacore call, got {len(hadacore)}")
    call_node = next(
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "hadacore_transform"
    )
    inplace = next(
        (kw.value for kw in call_node.keywords if kw.arg == "inplace"),
        None,
    )
    if not isinstance(inplace, ast.Constant) or inplace.value is not True:
        raise RuntimeError("Hadacore must use inplace=True and consume its return")
    if not any(
        isinstance(node, ast.Return) and node.value is call_node
        for node in ast.walk(helper)
    ):
        raise RuntimeError("helper does not return the Hadacore result")

    init = _function(model_tree, "Indexer", "__init__")
    init_text = ast.unparse(init)
    if "self.indexer_fp8_conditioning == 'none'" not in init_text:
        raise RuntimeError("fused Q path is not gated off while conditioning")

    forward = _function(model_tree, "Indexer", "forward")
    forward_calls = _calls(forward)
    condition_lines = [
        line
        for name, line in forward_calls
        if name == "_condition_indexer_fp8_row"
    ]
    quant_lines = [
        line for name, line in forward_calls if name == "per_token_group_quant_fp8"
    ]
    if len(condition_lines) != 2:
        raise RuntimeError("conditioning must be applied exactly once to Q and K")
    if len(quant_lines) != 1 or max(condition_lines) >= quant_lines[0]:
        raise RuntimeError("conditioning does not precede Q FP8 quantization")

    forward_text = ast.unparse(forward)
    q_concat = forward_text.find("q = torch.cat([q_pe, q_nope], dim=-1)")
    k_concat = forward_text.find("k = torch.cat([k_pe, k_nope], dim=-1)")
    conditioning = forward_text.find("q = _condition_indexer_fp8_row")
    if min(q_concat, k_concat, conditioning) < 0:
        raise RuntimeError("could not locate post-RoPE concatenation/conditioning")
    if conditioning <= max(q_concat, k_concat):
        raise RuntimeError("conditioning occurs before full-vector concatenation")

    return {
        "schema": "v20-indexer-wht-prototype-static-proof-v1",
        "config_sha256": _sha256(config_path),
        "model_sha256": _sha256(model_path),
        "checks": {
            "disabled_by_default": True,
            "server_config_field": True,
            "hadacore_return_consumed": True,
            "fused_q_disabled_when_conditioned": True,
            "post_rope_full_vector_order": True,
            "q_and_k_conditioned": True,
            "conditioning_before_fp8_quantization": True,
        },
        "verdict": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.config, args.model)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
