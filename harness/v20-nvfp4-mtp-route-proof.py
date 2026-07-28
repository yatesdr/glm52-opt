#!/usr/bin/env python3
"""CPU/source proof for the v20 compact-NVFP4 MTP route regression.

This proof deliberately does not import vLLM, Torch, CUDA, or SparkInfer.  It
checks the exact source trees used by the passing v19 and failing v20 images,
executes the patched route policy in isolation, proves the DCP position mapping
over the production range, and verifies the long-context top-k fold identity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
V19_SI = ROOT / "workspace/b12x-int8-v19-concurrency/b12x/attention"
V20_SI = ROOT / "workspace/sparkinfer-v20-review/sparkinfer/attention"
VLLM_REPO = ROOT / "workspace/gilded-gnosis"
DEFAULT_PATCHED_BACKEND = (
    ROOT
    / "workspace/vllm-v20-nvfp4-mtp-extend"
    / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
)
V19_COMMIT = "7ea567a2458a4800a6a0e3e0a6ba41fcbd00d146"
V20_COMMIT = "3e731bc043d23ec21277fb76d3e15fe6da91b23b"
BACKEND_PATH = "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"


def normalized_source(path: Path) -> str:
    text = path.read_text()
    replacements = (
        ("sparkinfer.attention._shared.mla", "stack.attention.mla"),
        ("b12x.attention.mla", "stack.attention.mla"),
        ("sparkinfer.attention.nsa_indexer", "stack.attention.indexer"),
        ("b12x.attention.indexer", "stack.attention.indexer"),
        ("sparkinfer", "stack"),
        ("b12x", "stack"),
        ("SPARKINFER", "STACK"),
        ("B12X", "STACK"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def find_symbol(tree: ast.AST, qualified_name: str) -> ast.AST:
    parts = qualified_name.split(".")
    nodes = list(getattr(tree, "body", ()))
    found: ast.AST | None = None
    for part in parts:
        found = next(
            (
                node
                for node in nodes
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.name == part
            ),
            None,
        )
        if found is None:
            raise AssertionError(f"missing symbol {qualified_name!r}")
        nodes = list(getattr(found, "body", ()))
    return found


def symbol_digest(path: Path, name: str) -> str:
    node = find_symbol(ast.parse(normalized_source(path)), name)
    payload = ast.dump(node, include_attributes=False).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_same_symbol(
    v19_path: Path, v20_path: Path, symbol: str
) -> tuple[str, str]:
    left = symbol_digest(v19_path, symbol)
    right = symbol_digest(v20_path, symbol)
    assert left == right, (
        f"{symbol} changed between v19 and v20: {left[:12]} != {right[:12]}"
    )
    return symbol, left[:12]


def git_source(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(VLLM_REPO), "show", f"{commit}:{path}"],
        text=True,
    )


def execute_patched_route_helper(patched_backend: Path):
    tree = ast.parse(patched_backend.read_text())
    helper = find_symbol(tree, "_resolve_spec_decode_mode")
    assert isinstance(helper, ast.FunctionDef)
    # Annotations are irrelevant to the pure policy and would otherwise require
    # importing the backend's module globals.
    helper.decorator_list = []
    helper.returns = None
    for arg in (*helper.args.posonlyargs, *helper.args.args, *helper.args.kwonlyargs):
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(patched_backend), "exec"), namespace)
    return namespace["_resolve_spec_decode_mode"]


def prove_route_delta(patched_backend: Path) -> dict[str, int]:
    v19 = git_source(V19_COMMIT, BACKEND_PATH)
    v20 = git_source(V20_COMMIT, BACKEND_PATH)
    assert 'VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "0"' in v19
    assert 'VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "auto"' in v20
    assert "use_spec_decode_kernel" not in v19
    assert "use_spec_decode_kernel" in v20

    route = execute_patched_route_helper(patched_backend)
    assert route("0", kv_cache_dtype="fp8_ds_mla") == (False, False)
    assert route("auto", kv_cache_dtype="fp8_ds_mla") == (True, False)
    assert route("auto", kv_cache_dtype="nvfp4_ds_mla") == (False, False)
    assert route("1", kv_cache_dtype="nvfp4_ds_mla") == (True, True)

    max_num_seqs = 16
    num_speculative_tokens = 3
    old_rows = max_num_seqs * (1 + num_speculative_tokens)
    new_rows = max_num_seqs
    heads = 64
    splits = 2048 // 64
    value_dim = 512
    row_delta = old_rows - new_rows
    tmp_output = row_delta * heads * splits * value_dim * 2
    tmp_lse = row_delta * heads * splits * 4
    output = row_delta * heads * value_dim * 2
    recovered = tmp_output + tmp_lse + output
    assert (old_rows, new_rows, splits) == (64, 16, 32)
    assert recovered > 95 * 1024 * 1024
    return {
        "old_rows": old_rows,
        "new_rows": new_rows,
        "tmp_output": tmp_output,
        "tmp_lse": tmp_lse,
        "output": output,
        "recovered": recovered,
    }


def prove_dcp_position_bijection() -> None:
    world = 4
    interleave = 1
    for global_pos in range(475_000):
        owner = (global_pos // interleave) % world
        local_pos = (
            (global_pos // (world * interleave)) * interleave
            + global_pos % interleave
        )
        reconstructed = (
            (local_pos // interleave) * world + owner
        ) * interleave + local_pos % interleave
        assert reconstructed == global_pos


def topk_pairs(items: list[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    return sorted(items, key=lambda item: (item[0], -item[1]), reverse=True)[:k]


def f32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", value))[0]


def monotone_f32_key(value: float) -> int:
    bits = f32_bits(value)
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def half_coarse_bin(value: float) -> int:
    bits16 = struct.unpack(">H", struct.pack(">e", value))[0]
    key16 = (~bits16 & 0xFFFF) if bits16 & 0x8000 else bits16 | 0x8000
    return (key16 >> 8) & 0xFF


def exact_radix_topk_indices(values: list[float], k: int) -> list[int]:
    """CPU form of v20's candidate-overflow fallback."""
    keys = [monotone_f32_key(value) for value in values]
    prefix = 0
    remaining = k
    for round_index in range(4):
        shift = 24 - round_index * 8
        mask = (0xFFFFFFFF << (32 - round_index * 8)) & 0xFFFFFFFF
        histogram = [0] * 256
        for key in keys:
            if round_index == 0 or (key & mask) == prefix:
                histogram[(key >> shift) & 0xFF] += 1
        count_greater = 0
        bucket = -1
        for candidate in range(255, -1, -1):
            count = histogram[candidate]
            if count_greater < remaining <= count_greater + count:
                bucket = candidate
                remaining -= count_greater
                break
            count_greater += count
        assert bucket >= 0
        prefix |= bucket << shift
    pivot = prefix
    greater = [index for index, key in enumerate(keys) if key > pivot]
    equal = [index for index, key in enumerate(keys) if key == pivot]
    selected = greater + equal[: k - len(greater)]
    assert len(selected) == k
    return selected


def prove_exact_overflow_fallback() -> None:
    # Put all values in the same FP16 coarse bucket so the threshold candidate
    # count greatly exceeds the CUDA path's 4,096-entry fast buffer.
    values = [
        struct.unpack(
            ">f",
            struct.pack(">I", f32_bits(0.125) + (index % 32_000)),
        )[0]
        for index in range(12_000)
    ]
    k = 2_048
    kth_bin = half_coarse_bin(
        sorted(values, reverse=True)[k - 1]
    )
    bin_count = sum(half_coarse_bin(value) == kth_bin for value in values)
    assert bin_count > 4_096
    selected = exact_radix_topk_indices(values, k)
    expected = sorted(
        range(len(values)),
        key=lambda index: (monotone_f32_key(values[index]), -index),
        reverse=True,
    )[:k]
    assert set(selected) == set(expected)


def prove_long_context_topk_fold() -> None:
    """TopK(TopK(A,k) union B,k) equals TopK(A union B,k)."""
    length = math.ceil(475_000 / 4)
    chunk = 32_768
    k = 2_048
    # Deterministic, unique FP32-like scores with the needle deliberately in the
    # oldest quarter.  The value construction avoids ties so set equality is
    # exact, not merely threshold-equivalent.
    needle = 17_321
    items = [
        (
            float(((index * 2_654_435_761) & 0xFFFFFFFF) - 0x80000000)
            / 2**31,
            index,
        )
        for index in range(length)
    ]
    items[needle] = (2.0, needle)
    direct = topk_pairs(items, k)
    folded: list[tuple[float, int]] = []
    for start in range(0, length, chunk):
        folded = topk_pairs(folded + items[start : start + chunk], k)
    assert folded == direct
    assert any(index == needle for _, index in folded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove the compact-NVFP4 MTP route correction."
    )
    parser.add_argument(
        "--patched-backend",
        type=Path,
        default=DEFAULT_PATCHED_BACKEND,
        help="b12x_mla_sparse.py containing the candidate route helper",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patched_backend = args.patched_backend.resolve()
    if not patched_backend.is_file():
        raise SystemExit(f"patched backend not found: {patched_backend}")

    identical = [
        assert_same_symbol(
            V19_SI / "mla/kv_cache.py",
            V20_SI / "_shared/mla/kv_cache.py",
            symbol,
        )
        for symbol in (
            "ConcatAndCacheNvfp4MlaFp8RopeKernel",
            "concat_and_cache_nvfp4_mla_fp8_rope",
        )
    ]
    identical.extend(
        assert_same_symbol(
            V19_SI / "mla/decode_math.py",
            V20_SI / "_shared/mla/decode_math.py",
            symbol,
        )
        for symbol in (
            "_nvfp4_pair_bfloat2",
            "s1_qk_nope_nvfp4_bf16",
            "s6_xv_nope_nvfp4_bf16",
        )
    )
    identical.extend(
        assert_same_symbol(
            V19_SI / "mla/prefill.py",
            V20_SI / "_shared/mla/prefill.py",
            "run_unified_prefill",
        )
        for _ in range(1)
    )
    identical.extend(
        assert_same_symbol(
            V19_SI / "indexer/paged.py",
            V20_SI / "nsa_indexer/paged.py",
            symbol,
        )
        for symbol in ("prepare_paged_indexer_metadata", "index_topk_fp8")
    )

    route = prove_route_delta(patched_backend)
    prove_dcp_position_bijection()
    prove_exact_overflow_fallback()
    prove_long_context_topk_fold()

    print("PASS: compact-NVFP4 KV writer/reader primitives are unchanged")
    for symbol, digest in identical:
        print(f"  {symbol}: {digest}")
    print(
        "PASS: the compared compact-NVFP4 route trees differ at the "
        "unqualified MTP decode auto-route"
    )
    print("PASS: patched route preserves fp8 auto and restores NVFP4 extend")
    print("PASS: DCP4 mapping is bijective through 475,000 positions")
    print("PASS: >4,096-candidate radix overflow fallback returns exact top-k")
    print("PASS: four-chunk 118,750-row top-k fold equals direct top-k")
    print(
        "scratch:",
        f"rows {route['old_rows']} -> {route['new_rows']}",
        f"minimum recovered={route['recovered'] / 2**20:.2f} MiB/GPU",
    )


if __name__ == "__main__":
    main()
