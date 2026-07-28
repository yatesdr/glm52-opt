# Proposed title

fix(glm52): preserve an explicit NCCL P2P level

```text
repo:   local-inference-lab/blackwell-llm-docker
base:   036706a7d769c35ad1a21083afca06fa31e11a8a
branch: fix/launcher-preserve-nccl-p2p-level
head:   9590e93
files:  2
```

## Summary

Keep `SYS` as the GLM-5.2 launcher default, but do not overwrite an explicit
`NCCL_P2P_LEVEL` supplied by the deployment:

```bash
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
```

Expose the resolved value in `DRY_RUN` output and make the release build gate
prove both contracts:

- no override resolves to `SYS`;
- an explicit `PXB` survives launcher processing.

## Why

The current launcher unconditionally exports `NCCL_P2P_LEVEL=SYS`. This
silently replaces a value supplied through Docker Compose, so container
inspection can show `PXB` even though the process actually runs with `SYS`.

CN4 has two PEX8747 switches in separate IIO domains. Its controlled fabric
matrix found that direct cross-switch P2P under `SYS` is pathological, while
`PXB` keeps direct traffic within a switch and uses the host path between
switches:

```text
packed-CKV gather: PXB was 10.1–11.3x faster than SYS
NCCL fallback:     PXB was approximately 8–11x faster
```

The default remains unchanged for systems that do not provide an override.
This is deployment-policy plumbing only; it does not alter vLLM, SparkInfer,
FlashInfer, CUDA kernels, checkpoint bytes, or model semantics.

## Validation

- `bash -n` passes for the launcher and release build script.
- `git diff --check` passes.
- The release build's existing image-level dry-run gate now asserts both the
  default and explicit-override cases.

CN4 production-candidate acceptance should additionally require the boot log
or process environment to report `NCCL_P2P_LEVEL=PXB`.
