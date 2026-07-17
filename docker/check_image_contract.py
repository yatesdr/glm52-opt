#!/usr/bin/env python3
"""Fail closed on drift in the 480k image build contract."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
SITE_PACKAGES = "/opt/venv/lib/python3.12/site-packages/"
BASE = (
    "davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv@"
    "sha256:99ae7b28bb7069b9f7a96f75ea815be56266d2cccf7808d4c497340bb8658bd5"
)

EXPECTED_ENV = {
    "PYTHONHASHSEED": "0",
    "HYBRID_TC_DECODE": "1",
    "HYBRID_NF3_TC_DECODE": "1",
    "HYBRID_HETERO_DECODE": "1",
    "HYBRID_TIER": "both",
    "HYBRID_KEPT": "b12x_nf3",
    "HYBRID_NF3": "b12x_nf3",
    "HYBRID_B12X_MAX_TOKENS": "2048",
    "HYBRID_MXFP8_NATIVE": "1",
    "B12X_EMPTY_CACHE_AFTER_WARMUP": "1",
    "VLLM_USE_AOT_COMPILE": "1",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "B12X_DCP_PREFILL_TRANSPORT": "ckv",
    "B12X_CKV_PHASE2_NCCL": "1",
    "B12X_CKV_HEADROOM_PROBES": "0",
    "B12X_DCP_GATHER_FP8": "0",
    "B12X_DCP_RS_RING": "0",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "CUTE_DSL_ARCH": "sm_120a",
    "OMP_NUM_THREADS": "16",
    "SAFETENSORS_FAST_GPU": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_LEVEL": "SYS",
    "NCCL_PROTO": "LL,LL128,Simple",
    "XDG_CACHE_HOME": "/cache/jit",
    "TRITON_CACHE_DIR": "/cache/jit/triton",
    "VLLM_USE_B12X_FP8_GEMM": "1",
    "VLLM_USE_B12X_MOE": "1",
    "VLLM_USE_B12X_SPARSE_INDEXER": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "B12X_MOE_FORCE_A16": "1",
    "VLLM_DCP_GLOBAL_TOPK": "1",
    "VLLM_DCP_SHARD_DRAFT": "1",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "1",
    "VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
    "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": "64KB",
    "B12X_DENSE_SPLITK_TURBO": "1",
    "B12X_MLA_SM120_PREFILL_F16_ROPE": "1",
    "VLLM_PCIE_DMA_FP8": "ag",
    "B12X_PCIE_DMA_FP8": "ag",
    "VLLM_USE_B12X_DCP_A2A": "1",
    "VLLM_DCP_A2A_MAX_TOKENS": "16",
    "VLLM_DCP_A2A_LARGE_BACKEND": "ag_rs",
    "VLLM_DCP_PROJECT_BEFORE_MERGE": "1",
    "VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS": "1024",
    "SWEEP_PROFILE_NAME": "v1_3_fast647_a2a16_dcp4_mnbt3072_mtp3",
}

EXPECTED_ARGV = shlex.split(
    r"""
    /model
    --served-model-name GLM-5.2
    --host 0.0.0.0 --port 5001
    --trust-remote-code
    --tensor-parallel-size 4
    --decode-context-parallel-size 4
    --dcp-comm-backend a2a
    --dcp-kv-cache-interleave-size 1
    --kv-cache-dtype nvfp4_ds_mla
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":56000000000,"secondary_tiers":[]}}'
    --attention-backend B12X_MLA_SPARSE
    --moe-backend b12x
    --load-format safetensors
    -cc.pass_config.fuse_allreduce_rms=True
    --gpu-memory-utilization 0.980
    --max-model-len 480000
    --max-num-seqs 8
    --max-num-batched-tokens 3072
    --num-gpu-blocks-override 2340
    --max-cudagraph-capture-size 32
    --async-scheduling
    --enable-chunked-prefill
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser glm47
    --reasoning-parser glm45
    --default-chat-template-kwargs '{"reasoning_effort":"high"}'
    --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'
    """
)

EXPECTED_ENTRYPOINT = [
    "/bin/sh",
    "-c",
    'rm -f /dev/shm/vllm_offload_*.mmap; exec vllm serve "$@"',
    "--",
]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instruction_json(text: str, name: str) -> list[str]:
    match = re.search(rf"(?m)^{name} (\[.*\])$", text)
    assert match is not None, f"missing JSON-form {name}"
    value = json.loads(match.group(1))
    assert isinstance(value, list) and all(isinstance(item, str) for item in value)
    return value


def main() -> None:
    text = DOCKERFILE.read_text()
    assert re.search(rf"(?m)^FROM {re.escape(BASE)}$", text), "base digest drift"

    manifest: dict[str, str] = {}
    for line in (ROOT / "docker/overlay-md5.txt").read_text().splitlines():
        expected_md5, destination = line.split(maxsplit=1)
        assert destination.startswith(SITE_PACKAGES)
        assert destination not in manifest, f"duplicate manifest path: {destination}"
        manifest[destination] = expected_md5

    copies: dict[str, Path] = {}
    for source, destination in re.findall(r"(?m)^COPY (\S+) (\S+)$", text):
        if not destination.startswith(SITE_PACKAGES):
            continue
        assert destination not in copies, f"duplicate image destination: {destination}"
        copies[destination] = ROOT / source

    assert len(manifest) == 12
    assert copies.keys() == manifest.keys(), "COPY set differs from 12-file manifest"
    for destination, source in copies.items():
        assert source.is_file(), f"missing COPY source: {source}"
        assert md5(source) == manifest[destination], f"MD5 drift: {source}"
        ast.parse(source.read_text(), filename=str(source))

    env_section = text.split("\nENV ", 1)[1].split("\n\n# Stale", 1)[0]
    env_tokens = shlex.split(env_section.replace("\\\n", " "))
    actual_env = dict(token.split("=", 1) for token in env_tokens)
    assert actual_env == EXPECTED_ENV, "production ENV drift"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in actual_env

    assert instruction_json(text, "ENTRYPOINT") == EXPECTED_ENTRYPOINT
    assert instruction_json(text, "CMD") == EXPECTED_ARGV, "production argv drift"

    compose = (ROOT / "docker-compose.yml").read_text()
    assert "shm_size: 64gb" in compose
    assert "ipc:" not in compose and "network_mode:" not in compose
    assert "64000" not in compose and "secondary_tiers" not in compose

    print("image contract: PASS (12 MD5s, base, env, entrypoint, argv, compose)")


if __name__ == "__main__":
    main()
