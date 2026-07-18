#!/usr/bin/env python3
"""CPU/source gate for the v18 368-byte FP8-RoPE writer bundle."""

from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "workspace/sol-v18-fp8-rope-writer"
WRITER = BUNDLE / "overlays/b12x/attention/mla/fp8_rope_writer.py"
LOADER_PATCH = BUNDLE / "vllm-loader.patch"
V13_WRITER = ROOT / "workspace/b12x-nf3-src/b12x/attention/mla/kv_cache.py"
V18_LOADER = (
    ROOT
    / "workspace/vllm-v18/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
)
CONTRIB_LOADER = (
    ROOT
    / "workspace/sol-v18-contributions/overlays/vllm/v1/attention/backends/mla"
    / "b12x_mla_sparse.py"
)
B12X_GIT = ROOT / "workspace/sol-workspace/repos/b12x-v17"
B12X_PIN = "bc85ef36192cb6e444d42ba7be86e1e125cca98a"
V18_B12X = ROOT / "workspace/v18-b12x-src/mla"


def _int_constants(path: Path) -> dict[str, int | float]:
    tree = ast.parse(path.read_text())
    expressions: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                expressions[target.id] = node.value

    values: dict[str, int | float] = {}

    def evaluate(name: str) -> int | float:
        if name in values:
            return values[name]
        node = expressions[name]

        def visit(expr: ast.expr) -> int | float:
            if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
                return expr.value
            if isinstance(expr, ast.Name):
                return evaluate(expr.id)
            if isinstance(expr, ast.BinOp):
                left, right = visit(expr.left), visit(expr.right)
                if isinstance(expr.op, ast.Add):
                    return left + right
                if isinstance(expr.op, ast.Sub):
                    return left - right
                if isinstance(expr.op, ast.Mult):
                    return left * right
                if isinstance(expr.op, ast.Div):
                    return left / right
                if isinstance(expr.op, ast.FloorDiv):
                    return left // right
            raise ValueError(f"unsupported expression for {name}: {ast.dump(expr)}")

        values[name] = visit(node)
        return values[name]

    for constant in expressions:
        try:
            evaluate(constant)
        except (KeyError, ValueError):
            pass
    return values


def _git_show(path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(B12X_GIT), "show", f"{B12X_PIN}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class WriterContractTests(unittest.TestCase):
    def test_v18_record_layout_is_complete_and_disjoint(self) -> None:
        constants = _int_constants(WRITER)
        self.assertEqual(constants["_NOPE_BYTES"], 256)
        self.assertEqual(constants["_SCALE_BYTES"], 32)
        self.assertEqual(constants["_ROPE_SCALE_OFFSET"], 288)
        self.assertEqual(constants["_PAD_OFFSET"], 292)
        self.assertEqual(constants["_PAD_BYTES"], 12)
        self.assertEqual(constants["_ROPE_OFFSET"], 304)
        self.assertEqual(constants["_RECORD_BYTES"], 368)

        ranges = [
            (0, 256, "latent"),
            (256, 288, "latent_scales"),
            (288, 292, "rope_scale"),
            (292, 304, "zero_pad"),
            (304, 368, "rope_e4m3"),
        ]
        cursor = 0
        for start, end, label in ranges:
            self.assertEqual(start, cursor, f"gap/overlap before {label}")
            self.assertGreater(end, start)
            cursor = end
        self.assertEqual(cursor, 368)

    def test_writer_stores_target_the_v18_offsets(self) -> None:
        source = WRITER.read_text()
        self.assertIn(
            "st_global_f32(dst + Int64(_ROPE_SCALE_OFFSET), rope_scale)", source
        )
        self.assertIn("dst + Int64(_PAD_OFFSET) + tid.to(Int64)", source)
        self.assertIn("st_global_u64(dst + Int64(_ROPE_OFFSET + 8 * w), q8)", source)
        self.assertIn('"_C_fp8_rope_ops::concat_and_cache_nvfp4_mla_fp8_rope"', source)
        self.assertIn('mutates_args=("kv_cache",)', source)
        self.assertIn("@_concat_and_cache_nvfp4_mla_fp8_rope_op.register_fake", source)

    def test_v13_numeric_recipe_is_preserved_but_layout_is_retargeted(self) -> None:
        old = V13_WRITER.read_text()
        new = WRITER.read_text()
        required_recipe = (
            "max_abs_16",
            "rcp_approx_ftz(Float32(6.0))",
            "cvt_f32_to_e4m3",
            "cvt_e4m3_to_f32_via_f16",
            "quantize_and_pack_16_fast",
            "fmax_f32",
            "fabs_f32",
            "Float32(_E4M3_MAX_RCP)",
            "cvt_f32x4_to_e4m3x4",
        )
        for expression in required_recipe:
            self.assertIn(expression, old)
            self.assertIn(expression, new)

        old_constants = _int_constants(V13_WRITER)
        new_constants = _int_constants(WRITER)
        self.assertEqual(old_constants["_RECORD_BYTES"], 368)
        self.assertEqual(old_constants["_ROPE_OFFSET"], 288)
        self.assertEqual(old_constants["_ROPE_SCALE_OFFSET"], 352)
        self.assertEqual(new_constants["_ROPE_OFFSET"], 304)
        self.assertEqual(new_constants["_ROPE_SCALE_OFFSET"], 288)

    def test_exact_b12x_reader_matches_writer_abi(self) -> None:
        io_source = (V18_B12X / "io.py").read_text()
        decode_source = (V18_B12X / "decode_math.py").read_text()
        self.assertIn("[288,304) holds fp32 scale + 12B zero pad", io_source)
        self.assertIn("[304,368) is E4M3", io_source)
        self.assertIn("_NVFP4_FP8_ROPE_GMEM_STRIDE = 368", io_source)
        self.assertIn(
            "scale = ld_shared_f32(kv_rope_base_addr + row_byte)", decode_source
        )
        self.assertIn("row_byte + Int32(16) + dim_even", decode_source)

    def test_mirrored_reader_is_the_exact_b12x_pin(self) -> None:
        for filename in ("io.py", "decode_math.py", "traits.py", "api.py"):
            mirrored = (V18_B12X / filename).read_bytes()
            pinned = _git_show(f"b12x/attention/mla/{filename}").encode()
            self.assertEqual(mirrored, pinned, filename)

    def test_exact_b12x_pin_exports_every_writer_dependency(self) -> None:
        fp4 = _git_show("b12x/cute/fp4.py")
        compiler = _git_show("b12x/cute/compiler.py")
        for helper in (
            "cvt_e4m3_to_f32_via_f16",
            "cvt_f32_to_e4m3",
            "cvt_f32x4_to_e4m3x4",
            "f16x2_to_f32x2",
            "fabs_f32",
            "fmax_f32",
            "get_ptr_as_int64",
            "max_abs_16",
            "quantize_and_pack_16_fast",
            "rcp_approx_ftz",
            "st_global_f32",
            "st_global_u64",
            "st_global_u8",
        ):
            self.assertIn(f"def {helper}", fp4)
        for helper in ("KernelCompileSpec", "tensor_compile_fact", "launch"):
            self.assertIn(helper, compiler)

    def test_v18_calls_exact_five_argument_schema(self) -> None:
        source = V18_LOADER.read_text()
        self.assertEqual(
            source.count(
                "torch.ops._C_fp8_rope_ops.concat_and_cache_nvfp4_mla_fp8_rope("
            ),
            2,
        )
        self.assertIn(
            "kv_c_normed,\n            k_pe_flat,\n            kv_cache,", source
        )
        self.assertIn(
            "kv_c,\n                k_pe_flat,\n"
            "                gathered_buffer,",
            source,
        )
        self.assertIn("slot_mapping.flatten(),\n            k_scale,", source)
        self.assertIn("slots,\n                k_scale,", source)

    def test_loader_prefers_source_and_fails_closed(self) -> None:
        source = LOADER_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in source.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("torch.ops.load_library(library)", source)
        self.assertIn('import_module("b12x.attention.mla.fp8_rope_writer")', source)
        self.assertIn("except ModuleNotFoundError as exc:", source)
        self.assertIn(
            'exc.name != "b12x.attention.mla.fp8_rope_writer"', source
        )
        self.assertIn("except Exception as exc:", source)
        self.assertLess(
            added.index('import_module("b12x.attention.mla.fp8_rope_writer")'),
            added.index("torch.ops.load_library(library)"),
        )
        self.assertIn("did not register the expected op", V18_LOADER.read_text())

    def test_loader_patch_applies_to_stock_and_current_contribution(self) -> None:
        for target in (V18_LOADER, CONTRIB_LOADER):
            result = subprocess.run(
                [
                    "patch",
                    "--dry-run",
                    "--batch",
                    "-p1",
                    "-d",
                    str(target.parents[5]),
                    "-i",
                    str(LOADER_PATCH),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
