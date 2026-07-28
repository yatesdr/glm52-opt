#!/usr/bin/env python3
"""Fail-closed tests for the shareable GLM-5.2 production entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "docker" / "serve-glm52-prod-auto.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


class ProdAutoEntrypointTest(unittest.TestCase):
    def run_wrapper(
        self,
        *,
        nccl_probe: str,
        nccl_static: str,
        wire_probe: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scripts = {
                "nccl": nccl_probe,
                "static": nccl_static,
                "wire": wire_probe,
            }
            paths: dict[str, Path] = {}
            for name, body in scripts.items():
                path = temp / f"{name}.py"
                _write_executable(path, body)
                paths[name] = path

            launcher = temp / "launcher.py"
            _write_executable(
                launcher,
                """\
                #!/usr/bin/env python3
                import json
                import os
                import sys

                keys = (
                    "NCCL_P2P_LEVEL",
                    "F8_DMA",
                    "SPARKINFER_PCIE_DMA_FP8",
                    "VLLM_PCIE_DMA_FP8",
                    "B12X_PCIE_DMA_FP8",
                    "DESTROYED_MXFP8",
                    "ONLINE_QUANT",
                    "QUANTIZATION_CONFIG_JSON",
                    "MAX_MODEL_LEN",
                )
                print(json.dumps({
                    "env": {key: os.environ.get(key) for key in keys},
                    "args": sys.argv[1:],
                }, sort_keys=True))
                """,
            )

            env = os.environ.copy()
            for key in (
                "NCCL_P2P_LEVEL",
                "F8_DMA",
                "SPARKINFER_PCIE_DMA_FP8",
                "VLLM_PCIE_DMA_FP8",
                "B12X_PCIE_DMA_FP8",
                "DESTROYED_MXFP8",
                "ONLINE_QUANT",
                "QUANTIZATION_CONFIG_JSON",
                "MAX_MODEL_LEN",
            ):
                env.pop(key, None)
            env.update(
                {
                    "GLM52_PYTHON": sys.executable,
                    "GLM52_NCCL_PROBE": str(paths["nccl"]),
                    "GLM52_NCCL_STATIC": str(paths["static"]),
                    "GLM52_WIRE_PROBE": str(paths["wire"]),
                    "GLM52_CANONICAL_LAUNCHER": str(launcher),
                    "XDG_CACHE_HOME": str(temp / "cache"),
                    "GPUS": "0,1,2,3",
                }
            )
            if extra_env:
                env.update(extra_env)

            return subprocess.run(
                ["/bin/bash", str(WRAPPER), "--example", "value"],
                check=False,
                capture_output=True,
                env=env,
                text=True,
            )

    def test_auto_uses_only_verified_probe_results(self) -> None:
        completed = self.run_wrapper(
            nccl_probe="print('PXB')",
            nccl_static="raise SystemExit(99)",
            wire_probe="print('i8_a2a')",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["env"]["NCCL_P2P_LEVEL"], "PXB")
        self.assertEqual(result["env"]["F8_DMA"], "i8_a2a")
        self.assertEqual(result["env"]["SPARKINFER_PCIE_DMA_FP8"], "i8_a2a")
        self.assertEqual(result["args"], ["--example", "value"])

    def test_failed_probes_choose_safe_uncompressed_fallback(self) -> None:
        completed = self.run_wrapper(
            nccl_probe="raise SystemExit(1)",
            nccl_static="raise SystemExit(1)",
            wire_probe="raise SystemExit(1)",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsNone(result["env"]["NCCL_P2P_LEVEL"])
        self.assertEqual(result["env"]["F8_DMA"], "0")
        self.assertIn("falling back to F8_DMA=0", completed.stderr)

    def test_explicit_values_skip_probes(self) -> None:
        completed = self.run_wrapper(
            nccl_probe="raise SystemExit(90)",
            nccl_static="raise SystemExit(91)",
            wire_probe="raise SystemExit(92)",
            extra_env={"NCCL_P2P_LEVEL": "PHB", "F8_DMA": "i8_ring"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["env"]["NCCL_P2P_LEVEL"], "PHB")
        self.assertEqual(result["env"]["F8_DMA"], "i8_ring")

    def test_destroyed_mode_never_changes_context(self) -> None:
        completed = self.run_wrapper(
            nccl_probe="raise SystemExit(90)",
            nccl_static="raise SystemExit(91)",
            wire_probe="raise SystemExit(92)",
            extra_env={
                "NCCL_P2P_LEVEL": "SYS",
                "F8_DMA": "i8",
                "DESTROYED_MXFP8": "1",
                "MAX_MODEL_LEN": "777777",
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["env"]["ONLINE_QUANT"], "custom")
        self.assertEqual(result["env"]["MAX_MODEL_LEN"], "777777")
        config = json.loads(result["env"]["QUANTIZATION_CONFIG_JSON"])
        self.assertEqual(config["linear"]["weight"], "mxfp8")
        self.assertNotIn("shared_experts", config)
        self.assertIn("consumes more VRAM", completed.stderr)

    def test_conflicting_wire_alias_fails_closed(self) -> None:
        completed = self.run_wrapper(
            nccl_probe="raise SystemExit(90)",
            nccl_static="raise SystemExit(91)",
            wire_probe="raise SystemExit(92)",
            extra_env={
                "NCCL_P2P_LEVEL": "SYS",
                "F8_DMA": "i8",
                "SPARKINFER_PCIE_DMA_FP8": "i8_ring",
            },
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("conflicting wire settings", completed.stderr)


if __name__ == "__main__":
    unittest.main()
