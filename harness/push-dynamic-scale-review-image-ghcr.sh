#!/usr/bin/env bash
# Push the ALREADY-BUILT dynamic-scale review image from CN4 to GHCR.
# Read-and-push only: touches no containers, no builds, no running work.
# Run ON CN4. Requires: docker login ghcr.io (user yatesdr + PAT with
# write:packages) done once beforehand.
set -euo pipefail

IMAGE_ID="sha256:db82fdcb5756d4a547853ba1330538bdd8a3dc0c6443c29bc49ba77b69b51cd1"
TAG="ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-nvfp4-dynamic-scale-review-20260728"

docker image inspect "$IMAGE_ID" --format 'found {{.Id}} size={{.Size}}' \
  || { echo "FATAL: gate-tested image not present on this host"; exit 1; }

docker tag "$IMAGE_ID" "$TAG"
docker push "$TAG"

echo "--- record these in the packaging handoff + both PR bodies ---"
docker image inspect "$TAG" --format 'ImageID: {{.Id}}'
docker manifest inspect "$TAG" -v 2>/dev/null | grep -m1 digest \
  || docker image inspect "$TAG" --format 'RepoDigests: {{.RepoDigests}}'
