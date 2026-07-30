#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="${GLM52_PUBLIC_TAG:-ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-r9-exl3-tr3-325-fused-source-rebuild-20260730}"

docker build \
  --pull=false \
  --progress=plain \
  --file "${repo_root}/docker/tr3-325-public/Dockerfile" \
  --tag "${tag}" \
  "${repo_root}"

docker image inspect "${tag}" \
  --format 'IMAGE_ID={{.Id}} BASE={{index .Config.Labels "org.opencontainers.image.base.digest"}} MODEL={{index .Config.Labels "ai.glm52.model.repository"}} MIXK={{index .Config.Labels "ai.glm52.exl3.output.sha256"}} PCIE_FUSION={{index .Config.Labels "ai.glm52.pcie.dma.fusion.output.sha256"}}'

docker push "${tag}"
docker image inspect "${tag}" --format 'LOCAL_IMAGE_ID={{.Id}}'
