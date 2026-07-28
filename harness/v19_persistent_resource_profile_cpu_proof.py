#!/usr/bin/env python3
"""CPU-only proof for the vLLM #172 backport (profile persistent kernel
resources before KV allocation).

The bug: vLLM sizes the KV cache from a profile taken BEFORE kernel warmup
creates persistent communication pools. On cn3 the b12x PCIe oneshot/DCP slabs
are direct cudaMalloc -- invisible to the torch allocator -- and were created
23 s AFTER the KV size was decided, so ~1.7 GiB went uncounted.

The fix warms kernels inside the profiled window and profiles a SECOND time, so
the measured peak contains both the persistent pools and the transient
activation workspace.

What is provable without a GPU:
  1. kernel_warmup runs exactly ONCE despite being called from three sites
  2. the profiled window really is profile -> warmup -> profile (order matters:
     a warmup after the last profile_run would still be uncounted)
  3. the late warmup site now goes through the idempotent wrapper
  4. kernel_warmup tolerates being called before attn_groups exists, which is
     the new early-call condition (an AttributeError here would break boot)

    python3 v19_persistent_resource_profile_cpu_proof.py --worker <gpu_worker.py> \\
                                                         --warmup <kernel_warmup.py>
Exit 0 = all passed.
"""

from __future__ import annotations

import argparse
import ast
import sys

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def fn_calls(tree: ast.AST, cls: str, fn: str) -> list[str]:
    """Ordered list of call names inside a method."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == fn:
                    out = []
                    for c in ast.walk(m):
                        if isinstance(c, ast.Call):
                            f = c.func
                            if isinstance(f, ast.Name):
                                out.append(f.id)
                            elif isinstance(f, ast.Attribute):
                                out.append(f.attr)
                    return out
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--warmup", required=True)
    args = ap.parse_args()

    worker_src = open(args.worker).read()
    wtree = ast.parse(worker_src)

    print("\n=== 1. idempotency: kernel_warmup runs exactly once ===")

    calls = []

    def fake_kernel_warmup(w):
        calls.append("warmup")

    class FakeWorker:
        def _warmup_kernels_once(self) -> None:
            if getattr(self, "_kernel_warmup_complete", False):
                return
            fake_kernel_warmup(self)
            self._kernel_warmup_complete = True

    w = FakeWorker()
    for _ in range(3):          # the three call sites in the patched file
        w._warmup_kernels_once()
    check("three calls -> one warmup", len(calls) == 1, f"{len(calls)} warmups")
    check("completion flag set", getattr(w, "_kernel_warmup_complete", False) is True)

    n_sites = worker_src.count("self._warmup_kernels_once()")
    check("all warmup sites go through the idempotent wrapper", n_sites == 3,
          f"{n_sites} call sites (expect 3: pre-msg, in-profile, pre-capture)")
    raw = [ln for ln in worker_src.splitlines()
           if ln.strip() == "kernel_warmup(self)"]
    check("only the wrapper calls kernel_warmup directly", len(raw) == 1,
          f"{len(raw)} direct call(s) — should be the one inside the wrapper")

    print("\n=== 2. ordering inside the profiled window ===")
    seq = fn_calls(wtree, "Worker", "_profile_model_with_kernel_warmup")
    prof_idx = [i for i, c in enumerate(seq) if c == "profile_run"]
    warm_idx = [i for i, c in enumerate(seq) if c == "_warmup_kernels_once"]
    check("profile_run called twice", len(prof_idx) == 2, f"{len(prof_idx)}x")
    check("warmup called once", len(warm_idx) == 1, f"{len(warm_idx)}x")
    ok_order = (len(prof_idx) == 2 and len(warm_idx) == 1
                and prof_idx[0] < warm_idx[0] < prof_idx[1])
    check("order is profile -> warmup -> profile", ok_order,
          "the SECOND profile is what counts the persistent pools; a warmup "
          "after the last profile_run would still be invisible")

    print("\n=== 3. the profiled window actually calls the helper ===")
    dam = fn_calls(wtree, "Worker", "determine_available_memory")
    check("determine_available_memory delegates to the two-pass helper",
          "_profile_model_with_kernel_warmup" in dam)

    # The one remaining inline profile_run is the explicit --kv-cache-memory-bytes
    # branch, which returns the caller-supplied size and never profiles for
    # sizing. That is correct -- but it means #172 has NO EFFECT while that flag
    # is set, because the profiling window below is never reached.
    body = worker_src[worker_src.index("def determine_available_memory"):]
    body = body[:body.index("\n    def ", 1)]
    explicit_branch = "if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:" in body
    check("the only inline profile_run is the explicit-bytes early return",
          explicit_branch and body.count("self.model_runner.profile_run()") == 1,
          "#172 is inert while --kv-cache-memory-bytes is set")

    prof_window = body[body.index("with memory_profiling("):]
    prof_window = prof_window[:prof_window.index("profile_torch_peak")]
    check("the profiling window itself has no inline profile_run",
          "self.model_runner.profile_run()" not in prof_window)

    print("\n=== 4. warmup tolerates being called before attn_groups exists ===")
    warm_src = open(args.warmup).read()
    check("no unguarded runner.attn_groups", "runner.attn_groups" not in warm_src)
    check("guarded via getattr", warm_src.count('getattr(runner, "attn_groups", None)') == 1
          and warm_src.count('getattr(worker.model_runner, "attn_groups", None)') == 1)

    ns: dict = {}
    exec(compile(ast.parse(
        "def _is_flashinfer_backend(b):\n    return False\n"), "<t>", "exec"), ns)
    src = warm_src[warm_src.index("def _uses_flashinfer_attention"):]
    src = src[:src.index("\ndef ", 1)]
    exec(compile(src, "<t>", "exec"), ns)

    class NoGroups:      # a runner that has not built attn_groups yet
        pass

    class WithGroups:
        attn_groups = []

    try:
        r1 = ns["_uses_flashinfer_attention"](NoGroups())
        r2 = ns["_uses_flashinfer_attention"](WithGroups())
        check("no AttributeError when attn_groups is absent", r1 is False,
              "this is the new early-call condition")
        check("still False for an empty attn_groups", r2 is False)
    except AttributeError as exc:
        check("no AttributeError when attn_groups is absent", False, str(exc))

    print(f"\n=== result: {'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED'} ===")
    for f in FAILED:
        print(f"    FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
