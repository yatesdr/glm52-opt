#!/usr/bin/env python3
"""Static gate for the rank-consistent block-INT8 AG overlay."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE_PY = REPO / "workspace/upstream-b12x-v19-kernel/b12x/distributed/pcie_dma.py"
BASE_CU = REPO / "workspace/upstream-b12x-v19-kernel/b12x/distributed/pcie_dma.cu"
OVERLAY_PY = ROOT / "overlays/b12x/distributed/pcie_dma.py"
OVERLAY_CU = ROOT / "overlays/b12x/distributed/pcie_dma.cu"
EXPECTED_BASE_PY_MD5 = "96e07e55c3843766999b88e184ce06dd"
EXPECTED_BASE_CU_MD5 = "356cff4d16db2364916325d369ea5fde"
EXPECTED_OVERLAY_PY_MD5 = "6cb81bcd74a7c1e43d62e6e623e34651"
EXPECTED_OVERLAY_CU_MD5 = "e30cd267a8dd6d4249f6298e5d13c6eb"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    if BASE_PY.exists():
        assert md5(BASE_PY) == EXPECTED_BASE_PY_MD5
    if BASE_CU.exists():
        assert md5(BASE_CU) == EXPECTED_BASE_CU_MD5

    python_source = OVERLAY_PY.read_text()
    cuda_source = OVERLAY_CU.read_text()
    assert md5(OVERLAY_PY) == EXPECTED_OVERLAY_PY_MD5
    assert md5(OVERLAY_CU) == EXPECTED_OVERLAY_CU_MD5
    ast.parse(python_source)

    assert 'if raw in ("i8", "int8", "ag_i8", "int8_ag")' in python_source
    assert "compressed_ag = compressed_eligible and self._fp8 in (" in python_source
    assert '(\n            "ag",\n            "ring",\n            "i8",\n        )' in python_source
    assert "int8_ag = compressed_eligible and self._fp8 == \"i8\"" in python_source
    assert "ext.dma_quant_i8 if int8_ag else ext.dma_quant" in python_source
    assert python_source.count("dequantize_ag(") == 2
    assert 'wire_mode = (\n            "int8-ag"' in python_source

    for token in (
        "quant_i8_kernel",
        "dequant_store_i8_kernel",
        'm.def("dma_quant_i8"',
        'm.def("dma_dequant_store_i8"',
    ):
        assert token in cuda_source

    # The INT8 mode must not enter the compressed reduce-scatter or a2a paths.
    assert 'self._fp8 == "a2a"' in python_source
    assert 'self._fp8 == "ring"' in python_source
    assert 'self._fp8 == "i8"' not in python_source[
        python_source.index("def _all_reduce_fp8") :
    ]

    print(
        f"PASS base_py={EXPECTED_BASE_PY_MD5} base_cu={EXPECTED_BASE_CU_MD5} "
        f"overlay_py={md5(OVERLAY_PY)} overlay_cu={md5(OVERLAY_CU)}"
    )


if __name__ == "__main__":
    main()
