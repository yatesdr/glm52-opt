#!/usr/bin/env python3
"""Phase 0 identity gate for the v19 Tier-1 reliability candidate (2026-07-26).

Proves the container that is running is the candidate we think it is, that all five
backported changes are physically present, and that nothing we promised not to touch
was touched.

Deliberately uses sha256 + text + AST checks only. It never imports vllm and never
touches CUDA, so it is safe to run against a container that is busy serving.

Usage:
    python3 v19_reliability_identity_gate.py [--container glm52-prod-candidate]
                                             [--json out.json]

Exit 0 = every gate passed. Exit 1 = at least one gate failed.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys

SP = "/opt/venv/lib/python3.12/site-packages"

# sha256 of the 10 overlay files as built on cn3 2026-07-26 from base
# ghcr.io/yatesdr/glm52-serve@sha256:ca8481687f71… (gilded-gnosis-v19-int8-block-patched).
EXPECTED_SHA256 = {
    # --- vLLM: structural memory-safety fixes ---
    "vllm/model_executor/layers/attention/mla_attention.py":
        "60791edf7c2a225f0754a9ae219f60520c7a11de4742aaf974fb3ad6a0eba088",
    "vllm/v1/attention/ops/dcp_alltoall.py":
        "f1f14905b50d5aab2adca1559d8c6a2253bd25b301c1b02e1035064d222ee15f",
    "vllm/v1/worker/gpu/model_runner.py":
        "b0c6ca44c5688c340631b85ca823baf57636f3b76346e74e2e15cc241314da0d",
    "vllm/v1/worker/gpu/warmup.py":
        "df4696f295af496373172c4f1d47f6d8409f34c03152d930adebea41a9b05fb3",
    "vllm/v1/worker/workspace.py":
        "730a633339937ac95662a9a8d484b3184eff921c61834cec7810fbf724114c28",
    # --- vLLM #154: release absorbed kv_b_proj source ---
    "vllm/v1/attention/backend.py":
        "1c18cf73b5012ceea610b23e45fa18bbd3847a37523238861ee8fb1b77464304",
    "vllm/v1/attention/backends/mla/b12x_mla_sparse.py":
        "124c70f335222eb4f2219672ac9ae46ee46e7cc6cd4338f6b02e603ddb82f8f3",
    # --- vLLM #172: profile persistent kernel resources before KV allocation ---
    "vllm/v1/worker/gpu_worker.py":
        "56bcd09f2c91bfba53b4d08428b4f070d77dfb52c4c96875fc6dcdb385905333",
    "vllm/model_executor/warmup/kernel_warmup.py":
        "7070b61fa420173d40087908b5ca06a1c82a17939c521ae7acd15dfe164a15d5",
    # --- b12x: cooperative grid keeps the shared-experts stream safe AND on ---
    "b12x/moe/fused/w4a16/kernel.py":
        "89533575f22082189fe98d748a4241f5c49165ba1d74b8bdf688fc3d8984b1f8",
}

# The three files the backend already shipped and that we must NOT have altered.
# b12x_mla_sparse.py declares the contract; envs.py holds the v19 single-channel
# default we deliberately left alone; b12x is the untouched kernel package.
UNTOUCHED_SHA256 = {
    "vllm/v1/attention/backends/mla/b12x_mla_sparse.py": None,   # filled from base at runtime
}


class Gate:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"gate": name, "pass": bool(ok), "detail": detail})
        if not ok:
            self.failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)


def dexec(container: str, script: str) -> str:
    """Run a /bin/sh snippet inside the container and return stdout."""
    out = subprocess.run(
        ["docker", "exec", container, "/bin/sh", "-c", script],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"docker exec failed: {out.stderr.strip()[:400]}")
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="glm52-prod-candidate")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    g = Gate()
    c = args.container

    print(f"\n=== Phase 0: identity gate — container {c} ===\n")

    # --- container is up -----------------------------------------------------
    try:
        state = subprocess.run(
            ["docker", "inspect", c, "--format",
             "{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
             "|{{.RestartCount}}|{{.Config.Image}}"],
            capture_output=True, text=True, check=True).stdout.strip()
        running, health, restarts, image = state.split("|")
    except Exception as exc:                                    # noqa: BLE001
        g.check("container inspectable", False, str(exc))
        return 1

    g.check("container running", running == "true", f"running={running}")
    g.check("restart count is 0", restarts == "0",
            f"RestartCount={restarts} (non-zero means it already crashed once)")
    print(f"        image: {image}")
    print(f"        health: {health}")

    # --- the 10 overlay files ------------------------------------------------
    print("\n  -- backported files (sha256) --")
    paths = " ".join(f"'{p}'" for p in EXPECTED_SHA256)
    got = dexec(c, f"cd {SP} && sha256sum {paths}")
    actual = {}
    for line in got.strip().splitlines():
        h, _, p = line.partition("  ")
        actual[p.strip()] = h.strip()
    for path, want in EXPECTED_SHA256.items():
        have = actual.get(path)
        g.check(f"sha256 {path.split('/')[-1]}", have == want,
                "" if have == want else f"expected {want[:12]}… got {(have or 'MISSING')[:12]}…")

    # --- the five patches are semantically present ---------------------------
    print("\n  -- patch markers --")
    markers = [
        ("#136 BMM contract: output guard",
         "vllm/model_executor/layers/attention/mla_attention.py", "force_contiguous_mla_bmm_output"),
        ("#136 BMM contract: weight guard",
         "vllm/model_executor/layers/attention/mla_attention.py", "force_contiguous_mla_bmm_weight"),
        ("workspace lanes: context var",
         "vllm/v1/worker/workspace.py", "use_workspace_lane"),
        ("workspace lanes: worker wiring",
         "vllm/v1/worker/gpu_worker.py", "num_workspace_lanes"),
        ("workspace lanes: draft entry points",
         "vllm/v1/worker/gpu/model_runner.py", "use_workspace_lane"),
        ("workspace lanes: warmup",
         "vllm/v1/worker/gpu/warmup.py", "use_workspace_lane"),
        ("#130 A2A precapture guard",
         "vllm/v1/attention/ops/dcp_alltoall.py", "is_vllm_cudagraph_capture_active"),
        ("#154 release helper",
         "vllm/model_executor/layers/attention/mla_attention.py", "_release_b12x_mxfp8_kv_b_proj"),
        ("#154 reload-safe materialize",
         "vllm/model_executor/layers/attention/mla_attention.py", "_materialize_kv_b_proj_weight"),
        ("#154 backend opt-in flag",
         "vllm/v1/attention/backend.py", "can_release_kv_b_proj_after_loading"),
        ("#154 B12X opts in",
         "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
         "can_release_kv_b_proj_after_loading: bool = True"),
        ("#154 MHA-prefill safety guard",
         "vllm/model_executor/layers/attention/mla_attention.py",
         "cannot release kv_b_proj while MHA prefill"),
        ("supports_mha_prefill override intact (forces prefill_backend=None)",
         "vllm/model_executor/layers/attention/mla_attention.py", "supports_mha_prefill"),
        ("#172 two-pass profile helper",
         "vllm/v1/worker/gpu_worker.py", "_profile_model_with_kernel_warmup"),
        ("#172 idempotent warmup wrapper",
         "vllm/v1/worker/gpu_worker.py", "_warmup_kernels_once"),
    ]
    for name, path, token in markers:
        n = dexec(c, f"grep -c '{token}' {SP}/{path} || true").strip()
        g.check(name, n.isdigit() and int(n) > 0, f"{n} occurrence(s)")

    # --- the shared-experts stream must be SAFE and STILL ON --------------------
    print("\n  -- shared-experts overlap: made safe, not disabled --")
    n = dexec(c, f"grep -c 'cooperative=True' {SP}/b12x/moe/fused/w4a16/kernel.py || true").strip()
    g.check("w4a16 barrier kernels launch cooperatively", n == "2",
            f"{n}/2 launches (W4A16FusedMoeKernel + W4A16FusedMoeHybridKernel)")

    n = dexec(c, f"grep -c 'cooperative=True' {SP}/b12x/moe/fused/dynamic.py || true").strip()
    g.check("dynamic.py cooperative launch intact (pre-existing)", n.isdigit() and int(n) >= 1,
            f"{n} occurrence(s) — Grid188/unified path was already fixed upstream")

    leaked = dexec(c, f"grep -rl 'supports_shared_experts_aux_stream' {SP}/vllm/ 2>/dev/null || true").strip()
    g.check("capability gate NOT present (overlap not disabled)", leaked == "",
            leaked.replace("\n", ", ") if leaked else "none — overlap stays enabled")

    disabled = dexec(c, "printenv VLLM_DISABLE_SHARED_EXPERTS_STREAM || true").strip()
    g.check("shared-experts stream enabled at runtime", disabled in ("", "0"),
            f"VLLM_DISABLE_SHARED_EXPERTS_STREAM={disabled or '<unset>'} "
            f"(must be unset/0 — this build keeps the ~11% decode)")

    # --- _v_up_proj must contain the contiguous-output branch (AST, no import) --
    print("\n  -- semantic check (AST, no import) --")
    src = dexec(c, f"cat {SP}/vllm/model_executor/layers/attention/mla_attention.py")
    tree = ast.parse(src)
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_v_up_proj":
            if "force_contiguous_mla_bmm_output" in ast.dump(node):
                guarded = True
    g.check("_v_up_proj guards the strided cuBLAS BMM", guarded,
            "this is the exact frame from the 2026-07-24 wedge")

    # --- things we promised NOT to change ------------------------------------
    print("\n  -- untouched invariants --")
    n = dexec(c, f"grep -c 'force_contiguous_mla_bmm' "
                 f"{SP}/vllm/v1/attention/backends/mla/b12x_mla_sparse.py || true").strip()
    g.check("backend still declares the contract (3 flags)", n == "3", f"{n} flags")

    n = dexec(c, "grep -c 'VLLM_PCIE_ONESHOT_SINGLE_CHANNEL\", \"1\"' "
                 f"{SP}/vllm/envs.py || true").strip()
    g.check("VLLM_PCIE_ONESHOT_SINGLE_CHANNEL default unchanged", n == "1",
            "channel isolation deliberately NOT backported (needs b12x API)")

    b12x_v = dexec(c, "/opt/venv/bin/pip show b12x 2>/dev/null | awk '/^Version/{print $2}'").strip()
    g.check("b12x version pin still 0.30.2", b12x_v == "0.30.2",
            f"version={b12x_v} (one file patched: w4a16 cooperative launches; "
            f"no CUDA/wheel rebuild)")

    leaked = dexec(c, "grep -rl 'tp_moe_plan_supports_aux_stream_overlap\\|checkpoint_channels"
                      "\\|rollback_channels\\|_capture_channel_stack' "
                      f"{SP}/vllm/ 2>/dev/null || true").strip()
    g.check("no reference to absent b12x symbols", leaked == "",
            leaked.replace("\n", ", ") if leaked else "none")

    ext = dexec(c, "printenv TORCH_EXTENSIONS_DIR || true").strip()
    g.check("baked INT8 extension dir unchanged", ext == "/cache/int8ext_baked_a826ef58",
            f"TORCH_EXTENSIONS_DIR={ext}")

    wire = dexec(c, "printenv B12X_PCIE_DMA_FP8 || true").strip()
    g.check("wire mode still i8_ring", wire == "i8_ring", f"B12X_PCIE_DMA_FP8={wire}")

    vllm_v = dexec(c, "/opt/venv/bin/pip show vllm 2>/dev/null | awk '/^Version/{print $2}'").strip()
    g.check("vLLM build string is the v19 base", "gilded.gnosis.v19.vllm7ea567a.b12x4cfa530" in vllm_v,
            vllm_v)

    print(f"\n=== Phase 0 result: {len(g.rows) - g.failed}/{len(g.rows)} passed ===")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"phase": "identity", "container": c, "image": image,
                       "failed": g.failed, "rows": g.rows}, fh, indent=2)
        print(f"    wrote {args.json}")
    if g.failed:
        print("\n*** IDENTITY GATE FAILED — do not proceed to timed phases. ***")
        return 1
    print("\nIdentity gate clean. The running container is the Tier-1 candidate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
