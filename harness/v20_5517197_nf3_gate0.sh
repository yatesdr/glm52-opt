#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'GATE0 FAIL: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -ge 1 && "$#" -le 2 ]] || die "usage: $0 IMAGE [SYS|PXB]"
image="$1"
p2p_level="${2:-SYS}"
[[ "${p2p_level}" == "SYS" || "${p2p_level}" == "PXB" ]] ||
  die "unsupported NCCL_P2P_LEVEL=${p2p_level}; expected SYS or PXB"

expected_base_manifest="sha256:e7a8a8549c10b5d16899e0fb45ff7eeca09dd7c1d1a83eee13fb03930d8eb80a"
expected_vllm="551719766029e78824a30d97ae6ac63917405b5f"
expected_sparkinfer="be0edcaae6f5d284bb29a82325aba7a0ead6960f"
expected_flashinfer="801d57a08958c13d375ddbb6be3be4808f48a708"
expected_b12x="a2002892614587a737475ef58834b9445a65de764bcbcd646c586a9162a2f2bf"
expected_launcher="fee02f8cd61a4c7edfc9d2b31b62f35ea18424ecde2968064eb212bd441fd883"

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')" ||
  die "image is not present"

LABELS="${labels}" python3 - \
  "${expected_base_manifest}" \
  "${expected_vllm}" \
  "${expected_sparkinfer}" \
  "${expected_flashinfer}" <<'PY'
import json
import os
import sys

labels = json.loads(os.environ["LABELS"])
expected = {
    "local-inference.base.manifest": sys.argv[1],
    "local-inference.vllm.commit": sys.argv[2],
    "local-inference.sparkinfer.commit": sys.argv[3],
    "local-inference.flashinfer.commit": sys.argv[4],
}
for key, value in expected.items():
    actual = labels.get(key)
    if actual != value:
        raise SystemExit(f"GATE0 FAIL: label {key}={actual!r}, expected {value!r}")
PY

hashes="$(
  docker run --rm --entrypoint /bin/sh "${image}" -c '
    sha256sum \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
      /usr/local/bin/serve-glm52-v16.sh |
      awk "{print \$1}"
  '
)" || die "cannot hash installed image files"
b12x_sha="$(sed -n '1p' <<<"${hashes}")"
launcher_sha="$(sed -n '2p' <<<"${hashes}")"
[[ -n "${b12x_sha}" && -n "${launcher_sha}" ]] ||
  die "image hash output is incomplete"

[[ "${b12x_sha}" == "${expected_b12x}" ]] ||
  die "b12x_mla_sparse.py=${b12x_sha}, expected ${expected_b12x}"
[[ "${launcher_sha}" == "${expected_launcher}" ]] ||
  die "serve-glm52-v16.sh=${launcher_sha}, expected ${expected_launcher}"

dry_run="$(
  docker run --rm \
    --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e DRY_RUN=1 \
    -e MODEL_FAMILY=glm52-hybrid \
    -e TP=4 \
    -e DCP=4 \
    -e MTP=3 \
    -e DCP_PREFILL_WORKSPACE=auto \
    -e MAX_MODEL_LEN=480000 \
    -e MAX_NUM_SEQS=16 \
    -e GRAPH=64 \
    -e KV_FP8_ROPE=1 \
    -e NCCL_P2P_LEVEL="${p2p_level}" \
    "${image}"
)"

require_line() {
  local line="$1"
  grep -Fqx "${line}" <<<"${dry_run}" ||
    die "dry-run line missing: ${line}"
}

require_text() {
  local text="$1"
  grep -Fq -- "${text}" <<<"${dry_run}" ||
    die "dry-run text missing: ${text}"
}

require_line "NCCL_P2P_LEVEL=${p2p_level}"
require_line "VLLM_DCP_QUERY_SPLIT=1"
require_line "VLLM_B12X_MLA_CKV_GATHER=1"
require_line "VLLM_DCP_TOPK_OWNER_MERGE=1"
require_line "VLLM_DCP_INDEXER_SHARDS=0"
require_line "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1"
require_line "VLLM_DCP_PROJECT_BEFORE_MERGE=1"
require_line "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1"
require_line "KV_CACHE_DTYPE=nvfp4_ds_mla"
require_line "QUANTIZATION=nvfp4_nf3_hybrid"
require_line "ONLINE_QUANT=nf3-mxfp8"
require_text "--max-model-len 480000"
require_text "--max-num-seqs 16"
require_text "--max-num-batched-tokens 2048"
require_text "--max-cudagraph-capture-size 64"
require_text "--speculative-config"

DRY_RUN="${dry_run}" python3 - <<'PY'
import json
import os
import shlex

command_line = next(
    (line for line in os.environ["DRY_RUN"].splitlines()
     if line.startswith("Command:")),
    None,
)
if command_line is None:
    raise SystemExit("GATE0 FAIL: dry-run Command line missing")
argv = shlex.split(command_line.removeprefix("Command:"))
try:
    index = argv.index("--speculative-config")
    config = json.loads(argv[index + 1])
except (ValueError, IndexError, json.JSONDecodeError) as error:
    raise SystemExit(f"GATE0 FAIL: cannot parse speculative config: {error}")
if config.get("num_speculative_tokens") != 3:
    raise SystemExit(
        "GATE0 FAIL: num_speculative_tokens="
        f"{config.get('num_speculative_tokens')!r}, expected 3"
    )
PY

printf '%s\n' "${dry_run}"
printf 'GATE0 PASS: exact 5517197 NF3 stack, pre-#171 route, rope8 request, %s routing, MTP3\n' "${p2p_level}"
