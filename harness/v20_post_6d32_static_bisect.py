#!/usr/bin/env python3
"""Fail-closed static bisect for the 6d32 -> fa71 needle regression.

The proof uses only source/config bytes already present in this repository. It
does not import vLLM, Torch, CUDA, or SparkInfer and does not contact a server.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OLD_REPO = ROOT / "workspace/vllm-v20-cn3-prod-proven"
NEW_REPO = ROOT / "workspace/vllm-v20-staged-bf16-fp8-query"
NO171_REPO = ROOT / "workspace/vllm-v20-staged-query-no171"
SPARK_REPO = ROOT / "workspace/sparkinfer-v20-current-recovery"
OLD_COMPOSE = ROOT / "deploy/glm52-prod-v20-promotion.yaml"
NEW_COMPOSE = ROOT / "deploy/glm52-v20-prod-ready-20260724.yaml"
BACKEND = Path("vllm/v1/attention/backends/mla/b12x_mla_sparse.py")
QUERY = Path("vllm/model_executor/kernels/attention/b12x_mxfp8_bmm.py")
SAFE_BMM = Path("csrc/libtorch_stable/attention/mla/safe_query_bmm.cu")
FUSED_INDEXER = Path("sparkinfer/attention/nsa_indexer/fused_indexer.py")
TILED_TOPK = Path("sparkinfer/attention/nsa_indexer/tiled_topk.py")
TOPK_PR_BODY = ROOT / "pr-v20-staged-bf16-fp8-query.md"
TOPK_LEDGER = ROOT / "v20-pr-ledger-20260724.md"
TOPK_ARTIFACT_SHA_PREFIX = "eb8b4e49"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    )


def _extract_function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path}: missing {name}")


def _function_ast_from_text(text: str, name: str) -> str:
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"source text: missing {name}")


def _execute_route_helper(repo: Path):
    path = repo / BACKEND
    helper = _extract_function(path, "_resolve_spec_decode_mode")
    helper.decorator_list = []
    helper.returns = None
    for arg in (*helper.args.posonlyargs, *helper.args.args, *helper.args.kwonlyargs):
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_resolve_spec_decode_mode"]


def _wire_mode(compose: Path) -> tuple[str, str]:
    text = compose.read_text()
    values = {}
    for name in ("SPARKINFER_PCIE_DMA_FP8", "VLLM_PCIE_DMA_FP8"):
        match = re.search(rf"{name}(?:=|:\s*)[\"']?([a-zA-Z0-9_]+)", text)
        assert match, f"{compose}: missing {name}"
        values[name] = match.group(1)
    return values["SPARKINFER_PCIE_DMA_FP8"], values["VLLM_PCIE_DMA_FP8"]


def _documented_topk_evidence() -> dict[str, str | int]:
    """Validate the checked-in record without misrepresenting its provenance.

    The successful JSONL lives on the operator host, not in this repository.
    A similarly named local JSONL is an earlier failed invocation and must not
    be accepted as the 160/160 artifact. The static bisect therefore verifies
    the checked-in report and its artifact pin, and labels the evidence
    historical rather than claiming to have replayed the GPU proof locally.
    """

    pr_body = TOPK_PR_BODY.read_text()
    ledger = TOPK_LEDGER.read_text()
    assert "160/160 exact" in pr_body
    assert "top-k proof passed 160/160" in ledger
    assert TOPK_ARTIFACT_SHA_PREFIX in pr_body
    assert TOPK_ARTIFACT_SHA_PREFIX in ledger
    return {
        "passed": 160,
        "failed": 0,
        "provenance": "documented external GPU artifact",
        "sha256_prefix": TOPK_ARTIFACT_SHA_PREFIX,
    }


def main() -> None:
    old_wire = _wire_mode(OLD_COMPOSE)
    new_wire = _wire_mode(NEW_COMPOSE)
    assert old_wire == new_wire == ("i8_ring", "i8_ring")

    # The new safe op deliberately changed its cuBLAS compute mode.
    old_safe = _git(NEW_REPO, "show", f"af9d01cf1:{SAFE_BMM}")
    new_safe = _git(NEW_REPO, "show", f"992b874cf:{SAFE_BMM}")
    assert "CUBLAS_COMPUTE_32F_PEDANTIC" in old_safe
    assert "CUBLAS_COMPUTE_32F_PEDANTIC" not in new_safe
    assert "CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT" in new_safe

    # The staged guard disables only the SparkInfer fused BF16 query when the
    # requested output is FP8. It does not restore the compiled safe-BMM mode.
    query_text = (NEW_REPO / QUERY).read_text()
    assert "if output_dtype != torch.bfloat16:" in query_text
    assert "return False" in query_text
    new_compose = NEW_COMPOSE.read_text()
    assert re.search(r"VLLM_B12X_ABSORB_BMM(?:=|:\s*)[\"']?0", new_compose)

    current_route = _execute_route_helper(NEW_REPO)
    assert current_route("auto", kv_cache_dtype="fp8_ds_mla") == (True, False)
    assert current_route("auto", kv_cache_dtype="nvfp4_ds_mla") == (
        False,
        False,
    )

    # The prepared no-#171 branch retains every later memory/query change but
    # restores the pre-#171 route implementation: auto uses decode.
    assert _git(NO171_REPO, "rev-parse", "HEAD").strip() == (
        "b8534c4a5fad9500c0aebd0ef5e293672033fc4e"
    )
    no171_text = (NO171_REPO / BACKEND).read_text()
    assert "_resolve_spec_decode_mode" not in no171_text
    assert "self.spec_extend_as_decode = spec_decode_mode not in disabled_modes" in (
        no171_text
    )

    topk_evidence = _documented_topk_evidence()
    fused_indexer = (SPARK_REPO / FUSED_INDEXER).read_text()
    old_tiled_topk = _git(SPARK_REPO, "show", f"ffa922b0:{TILED_TOPK}")
    new_tiled_topk = (SPARK_REPO / TILED_TOPK).read_text()
    assert "crossover = max(crossover, 16384)" in fused_indexer
    assert "from sparkinfer.attention.nsa_indexer.tiled_topk import (" in (
        fused_indexer
    )
    assert "_SMEM_CANDS," in fused_indexer
    # _SMEM_CANDS appears only in the import list; the fused kernel sizes its
    # candidate buffers from carry_cap instead. Do not mislabel the tiled
    # selector's 4096 -> 8192 change as a fused-kernel byte delta.
    assert fused_indexer.count("_SMEM_CANDS") == 1
    assert "_SMEM_CANDS = 4096" in old_tiled_topk
    assert "_SMEM_CANDS = 8192" in new_tiled_topk
    assert (
        _git(
            SPARK_REPO,
            "diff",
            "--exit-code",
            "ffa922b0c06e5c45ed1344bdc5260cc9c7e85c9a",
            "a93df671cc7b33734f499b57228e542c3d3c3697",
            "--",
            str(FUSED_INDEXER),
        )
        == ""
    )
    for helper in (
        "_convert_to_uint8",
        "_convert_to_uint32",
        "_exact_overflow_fallback",
    ):
        assert _function_ast_from_text(
            old_tiled_topk, helper
        ) == _function_ast_from_text(new_tiled_topk, helper)
    # Four GLM query rows on a 188-SM GPU use 47 CTAs per group. The fitted
    # threshold is below the explicit GLM floor, so the live runtime crossover
    # is exactly 16,384 DCP-local tokens == 65,536 global DCP4 tokens.
    ctas_per_group = 188 // 4
    fitted = 22000 + 117 * ctas_per_group - 13 * (2048 - 512)
    fused_crossover = max(16384, fitted)
    assert (ctas_per_group, fitted, fused_crossover) == (47, 7531, 16384)
    result = {
        "status": "PASS",
        "wire_control": {
            "old": old_wire,
            "new": new_wire,
            "conclusion": "same i8_ring wire; wire cannot explain the onset shift",
        },
        "topk_control": {
            **topk_evidence,
            "conclusion": (
                "tiled selector cleared; this does not cover the fused "
                "cooperative merge arm"
            ),
        },
        "candidate_1": {
            "change": "#171 NVFP4 verifier route",
            "old": "auto -> split-K decode",
            "new": "auto -> single-pass extend",
            "coverage_gap": (
                "routing tests only; numerical causality test is fp8_ds_mla, "
                "not nvfp4_ds_mla/DCP4"
            ),
            "no171_head": _git(NO171_REPO, "rev-parse", "--short=12", "HEAD").strip(),
        },
        "candidate_2": {
            "change": "safe_mla_query_bmm compute mode",
            "old": "CUBLAS_COMPUTE_32F_PEDANTIC",
            "new": "CUBLAS_COMPUTE_32F",
            "coverage_gap": (
                "current test permits rtol=atol=0.05 and does not compare "
                "post-quantization bytes or selected retrieval ids"
            ),
        },
        "candidate_3": {
            "change": "latent fused GLM merge-arm interaction",
            "serial_local_max": fused_crossover,
            "cooperative_local_min": fused_crossover + 1,
            "global_dcp4_boundary": fused_crossover * 4,
            "field_observation": "60k marginal; 70k+ deterministic miss",
            "source_delta": (
                "none between ffa922b and a93df671; imported selector helpers "
                "used by fused are AST-identical and _SMEM_CANDS is unused"
            ),
            "coverage_gap": (
                "160/160 artifact exercises tiled top-k, not production "
                "32-head/topk-2048 fused cooperative graph replay"
            ),
            "conclusion": (
                "the exact field boundary makes this a high-value interaction "
                "discriminator, but the merge arm alone cannot explain the "
                "post-6d32 onset shift"
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
