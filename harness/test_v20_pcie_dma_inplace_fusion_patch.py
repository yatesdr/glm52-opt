#!/usr/bin/env python3
"""CPU/source gates for the narrow B12X PCIe DMA in-place fusion patch."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker" / "tr3-325-public" / "patch_pcie_dma_inplace_fusion.py"


def load_patcher_constants() -> tuple[str, str, str]:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    constants: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            constants[node.targets[0].id] = node.value.value
    return constants["OLD"], constants["NEW"], constants["MARKER"]


class InplaceFusionPatchTest(unittest.TestCase):
    def test_patch_is_fail_closed_and_idempotent(self) -> None:
        old, new, marker = load_patcher_constants()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom_all_reduce.py"
            target.write_text(old, encoding="utf-8")
            env = dict(os.environ, VLLM_CUSTOM_ALLREDUCE_PATH=str(target))
            first = subprocess.run(
                [sys.executable, str(PATCHER)],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), new)
            self.assertEqual(new.count(marker), 1)

            second = subprocess.run(
                [sys.executable, str(PATCHER)],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already patched", second.stdout)

    def test_generic_dma_default_semantics_are_not_patched(self) -> None:
        _, new, _ = load_patcher_constants()
        self.assertIn("self._pcie_dma.all_reduce(inp, out=inp)", new)
        self.assertNotIn("out = torch.empty_like(inp)", new)
        self.assertIn("ops.fused_add_rms_norm", new)
        self.assertIn("if dma_eligible:", new)

    def test_shape_and_alias_checks_precede_dma_dispatch(self) -> None:
        _, new, _ = load_patcher_constants()
        validation = new.index("if (\n            inp.ndim == 0")
        dispatch = new.index("if dma_eligible:")
        self.assertLess(validation, dispatch)
        self.assertIn("inp.data_ptr() == residual.data_ptr()", new)


if __name__ == "__main__":
    unittest.main()
