#!/usr/bin/env python3
"""GPU/Docker-free proof for the latest-head v20 derived images."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROD_TREE = ROOT / "workspace/vllm-v20-prod-integration-head"
PROF_TREE = ROOT / "workspace/vllm-v20-prod-profiler-head"
SI_TREE = ROOT / "workspace/sparkinfer-v20-current-recovery"
PROD_DOCKERFILE = ROOT / "docker/Dockerfile.v20-prod-head-20260725"
PROF_DOCKERFILE = ROOT / "docker/Dockerfile.v20-prod-profiler-head-20260725"
VLLM_PROD_MANIFEST = (
    ROOT / "patches/v20-prod-head/manifests/vllm-production.sha256"
)
VLLM_PROF_MANIFEST = (
    ROOT / "patches/v20-prod-head/manifests/vllm-profiler.sha256"
)
SI_MANIFEST = (
    ROOT / "patches/v20-prod-head/manifests/sparkinfer-production.sha256"
)
VLLM_CONTAINER_BASE = "992b874cf7ae504616bbb1d2d4f7a7355be6972b"
VLLM_PROD_HEAD = "625ac3b75bf26741b4d8de06a46ec803a8a80f23"
VLLM_PROF_HEAD = "71053e516c13279d5735a54431cb44a8111d4af3"
SI_CONTAINER_BASE = "a93df671cc7b33734f499b57228e542c3d3c3697"
SI_HEAD = "d4969d993cdd16cc417056d471af42d10ac3fada"
MANIFEST_HASHES = {
    VLLM_PROD_MANIFEST: (
        "107e471dd587b0eb916598af3a1cd51b8d315393863f3f1410cdca67b48a25fb"
    ),
    VLLM_PROF_MANIFEST: (
        "440ad3f6cf2e52b4d42d7d519fc0b9f1c6fac530a634d4220a35d58145acb59b"
    ),
    SI_MANIFEST: (
        "cf7e42e38bc21cb52e87a177b9109e8ab75ec6158990b271b4dc5a6f18400a43"
    ),
}


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(tree), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest(tree: Path, manifest: Path, root_name: str) -> int:
    expected_paths = set(_git(tree, "ls-files", root_name).splitlines())
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip()
        assert relative.startswith(f"{root_name}/")
        assert relative not in seen
        seen.add(relative)
        assert _sha(tree / relative) == digest, relative
    assert seen == expected_paths
    return len(seen)


def main() -> int:
    assert _git(PROD_TREE, "rev-parse", "HEAD").strip() == VLLM_PROD_HEAD
    assert _git(PROF_TREE, "rev-parse", "HEAD").strip() == VLLM_PROF_HEAD
    assert _git(PROF_TREE, "rev-parse", "HEAD^").strip() == VLLM_PROD_HEAD
    assert _git(SI_TREE, "rev-parse", "HEAD").strip() == SI_HEAD
    for tree in (PROD_TREE, PROF_TREE, SI_TREE):
        subprocess.run(("git", "-C", str(tree), "diff", "--quiet"), check=True)
        subprocess.run(
            ("git", "-C", str(tree), "diff", "--cached", "--quiet"),
            check=True,
        )

    # The published image's stable extension remains valid only because every
    # package delta copied over it is Python/Triton source.
    vllm_delta = _git(
        PROD_TREE,
        "diff",
        "--name-only",
        f"{VLLM_CONTAINER_BASE}..{VLLM_PROD_HEAD}",
        "--",
        "vllm",
    ).splitlines()
    si_delta = _git(
        SI_TREE,
        "diff",
        "--name-only",
        f"{SI_CONTAINER_BASE}..{SI_HEAD}",
        "--",
        "sparkinfer",
    ).splitlines()
    assert vllm_delta and all(path.endswith(".py") for path in vllm_delta)
    assert si_delta and all(path.endswith(".py") for path in si_delta)

    for manifest, expected in MANIFEST_HASHES.items():
        assert _sha(manifest) == expected
    vllm_prod_count = _verify_manifest(
        PROD_TREE,
        VLLM_PROD_MANIFEST,
        "vllm",
    )
    vllm_prof_count = _verify_manifest(
        PROF_TREE,
        VLLM_PROF_MANIFEST,
        "vllm",
    )
    si_count = _verify_manifest(SI_TREE, SI_MANIFEST, "sparkinfer")
    assert vllm_prof_count == vllm_prod_count + 1

    prod = PROD_DOCKERFILE.read_text(encoding="utf-8")
    prof = PROF_DOCKERFILE.read_text(encoding="utf-8")
    for value in (
        VLLM_CONTAINER_BASE,
        VLLM_PROD_HEAD,
        SI_CONTAINER_BASE,
        SI_HEAD,
        MANIFEST_HASHES[VLLM_PROD_MANIFEST],
        MANIFEST_HASHES[SI_MANIFEST],
    ):
        assert value in prod
    assert "compute_phase_profiler.py" in prod
    assert "test ! -e vllm/model_executor/layers/compute_phase_profiler.py" in prod
    assert 'io.yatesdr.diagnostics="none"' in prod
    assert VLLM_PROF_HEAD in prof and VLLM_PROD_HEAD in prof
    assert MANIFEST_HASHES[VLLM_PROF_MANIFEST] in prof
    assert 'io.yatesdr.promotable="false"' in prof
    assert "test -f vllm/model_executor/layers/compute_phase_profiler.py" in prof

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for directory in (
        "workspace/vllm-v20-prod-integration-head/vllm/**",
        "workspace/vllm-v20-prod-profiler-head/vllm/**",
        "workspace/sparkinfer-v20-current-recovery/sparkinfer/**",
    ):
        assert f"!{directory}" in dockerignore
    assert "**/__pycache__/" in dockerignore
    assert "**/*.py[cod]" in dockerignore

    print(
        "PASS v20 packaging: source-only ABI delta, complete manifests "
        f"(vllm={vllm_prod_count}, profiler={vllm_prof_count}, "
        f"sparkinfer={si_count}), clean production image, non-promotable "
        "profiler child"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
