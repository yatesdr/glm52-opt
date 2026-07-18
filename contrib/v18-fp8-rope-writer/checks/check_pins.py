#!/usr/bin/env python3
"""Verify v18 writer source, reader, patch, proof, and handoff byte pins."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "workspace"
BUNDLE = WORKSPACE / "sol-v18-fp8-rope-writer"
B12X_GIT = WORKSPACE / "sol-workspace/repos/b12x-v17"
B12X_PIN = "bc85ef36192cb6e444d42ba7be86e1e125cca98a"

FILE_PINS = {
    WORKSPACE / "b12x-nf3-src/b12x/attention/mla/kv_cache.py": (
        "29e2501a661a1e07634e3bc4d99ba7da"
    ),
    WORKSPACE / "vllm-v18/vllm/v1/attention/backends/mla/b12x_mla_sparse.py": (
        "14c14eabc937cddf481532fb19e1dcb5"
    ),
    WORKSPACE / "v18-b12x-src/mla/io.py": "0a961cc5df4e79c282f61d70f35bb4ea",
    WORKSPACE / "v18-b12x-src/mla/decode_math.py": (
        "b25ffccf04573e33dfc8dbf64cc68725"
    ),
    WORKSPACE / "v18-b12x-src/mla/traits.py": (
        "6497ca8197d3749fa796cf267f2c3519"
    ),
    WORKSPACE / "v18-b12x-src/mla/api.py": "58a3deb6478aad90f989b19a5304e7af",
    BUNDLE / "overlays/b12x/attention/mla/fp8_rope_writer.py": (
        "f616822c81a2b57f6a7de2dbf1da55ab"
    ),
    BUNDLE / "vllm-loader.patch": "09838476add9b577fbdc38263e45cfd3",
    BUNDLE / "checks/test_writer_contract.py": (
        "1c7caf8b2439970b6feb11cc0c940ae3"
    ),
    BUNDLE / "checks/gpu_writer_smoke.py": (
        "0d2c0f6139ecfe88eee0f9a2da9e1086"
    ),
    BUNDLE / "checks/gpu_writer_reader_roundtrip.py": (
        "0cbbdc93f06362236e5f0484e0a6a357"
    ),
    BUNDLE / "README.md": "0981fba3fbe16398295ad28c5a66036c",
    BUNDLE / "MANIFEST.md": "9c6e6f06e1fc295803f976dde8c04b3c",
}

GIT_BLOB_PINS = {
    "b12x/attention/mla/io.py": "0a961cc5df4e79c282f61d70f35bb4ea",
    "b12x/attention/mla/decode_math.py": "b25ffccf04573e33dfc8dbf64cc68725",
    "b12x/cute/fp4.py": "43d10f02b9b79cb4292ed3aa43feb44f",
    "b12x/cute/compiler.py": "f1a3ce0d0e501b60c84d21d1c0da6b0e",
    "b12x/attention/mla/traits.py": "6497ca8197d3749fa796cf267f2c3519",
    "b12x/attention/mla/api.py": "58a3deb6478aad90f989b19a5304e7af",
}

CONTRIB = (
    WORKSPACE
    / "sol-v18-contributions/overlays/vllm/v1/attention/backends/mla"
    / "b12x_mla_sparse.py"
)
CONTRIB_STATES = {
    "715e0c0d385e873ef87ef86f70ba3a53": "unpatched input",
    "a126a1b93a5195c29458a105578f8c4b": "expected patched output",
}


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path, expected in FILE_PINS.items():
        actual = _md5_file(path)
        assert actual == expected, f"md5 mismatch: {path}: {actual} != {expected}"
        print(f"PASS {expected}  {path.relative_to(ROOT)}")

    contrib_md5 = _md5_file(CONTRIB)
    assert contrib_md5 in CONTRIB_STATES, (
        f"unknown combined-overlay state: {CONTRIB}: {contrib_md5}"
    )
    print(
        f"PASS {contrib_md5}  {CONTRIB.relative_to(ROOT)} "
        f"({CONTRIB_STATES[contrib_md5]})"
    )

    for path, expected in GIT_BLOB_PINS.items():
        data = subprocess.run(
            ["git", "-C", str(B12X_GIT), "show", f"{B12X_PIN}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
        actual = _md5_bytes(data)
        assert actual == expected, (
            f"git blob md5 mismatch: {path}: {actual} != {expected}"
        )
        print(f"PASS {expected}  {B12X_PIN}:{path}")


if __name__ == "__main__":
    main()
