#!/usr/bin/env python3
"""Patch v20-r9's B12X fused AR+RMS fallback to use DMA in-place.

The generic PCIe DMA API intentionally remains out-of-place by default:
callers may retain one result while a later collective runs.  The fused
all-reduce + residual-add RMSNorm custom op has a narrower lifetime contract:
its existing fallback immediately copies the all-reduce result back over the
input and discards the temporary.  For a production 3072 x 6144 BF16 prefill,
that temporary is a late 36 MiB allocation after the KV pool is live.

When the already-selected PCIe DMA backend accepts the tensor, reduce directly
into that input and run the same fused add/RMSNorm operation.  The production
geometry and i8_ring mode are bit-proven against the out-of-place path by
harness/v20_pcie_dma_inplace_equivalence_proof.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MARKER = "# === B12X-PCIE-DMA-INPLACE-FUSION v1 ==="


def find_target() -> Path:
    override = os.environ.get("VLLM_CUSTOM_ALLREDUCE_PATH")
    if override:
        return Path(override)
    import vllm

    return (
        Path(vllm.__file__).parent
        / "distributed"
        / "device_communicators"
        / "custom_all_reduce.py"
    )


OLD = '''    def try_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> bool:
        """Run the B12X fused collective when all operands are supported."""
        inp_size = inp.numel() * inp.element_size()
        if (
            not self.supports_fused_add_rms_norm()
            or self._pcie_fused_add_rms_norm_max_size is None
            or inp_size > self._pcie_fused_add_rms_norm_max_size
        ):
            return False
        assert self._pcie_runtime is not None
        if not self._pcie_runtime.for_stream(
            self._pcie_runtime_stream()
        ).should_allreduce(inp):
            return False
        if (
            inp.ndim == 0
            or residual.shape != inp.shape
            or residual.dtype != inp.dtype
            or residual.device != inp.device
            or not is_weak_contiguous(residual)
            or weight.shape != (inp.shape[-1],)
            or weight.dtype != inp.dtype
            or weight.device != inp.device
            or not weight.is_contiguous()
            or inp.shape[-1] * inp.element_size() % 16 != 0
            or inp.data_ptr() == residual.data_ptr()
            or epsilon < 0
        ):
            return False

        self._pcie_runtime.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=inp,
            residual_out=residual,
            stream=self._pcie_runtime_stream(),
        )
        return True
'''


NEW = '''    def try_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> bool:
        """Run the B12X fused or allocation-free DMA + RMSNorm path."""
        inp_size = inp.numel() * inp.element_size()
        runtime_stream = self._pcie_runtime_stream()
        oneshot_eligible = (
            self.supports_fused_add_rms_norm()
            and self._pcie_fused_add_rms_norm_max_size is not None
            and inp_size <= self._pcie_fused_add_rms_norm_max_size
            and self._pcie_runtime is not None
            and self._pcie_runtime.for_stream(runtime_stream).should_allreduce(inp)
        )
        dma_eligible = (
            self._pcie_dma is not None
            and self._pcie_dma.should_allreduce(inp)
        )
        if not oneshot_eligible and not dma_eligible:
            return False
        if (
            inp.ndim == 0
            or residual.shape != inp.shape
            or residual.dtype != inp.dtype
            or residual.device != inp.device
            or not is_weak_contiguous(residual)
            or weight.shape != (inp.shape[-1],)
            or weight.dtype != inp.dtype
            or weight.device != inp.device
            or not weight.is_contiguous()
            or inp.shape[-1] * inp.element_size() % 16 != 0
            or inp.data_ptr() == residual.data_ptr()
            or epsilon < 0
        ):
            return False

        if dma_eligible:
            # === B12X-PCIE-DMA-INPLACE-FUSION v1 ===
            # This custom op immediately overwrites inp with the all-reduce
            # result and consumes it in RMSNorm.  Writing the ring result
            # directly to inp preserves that exact lifetime while avoiding
            # torch.empty_like(inp) after KV-cache sizing.  Do not move this
            # behavior into PCIeDmaAllReduce's generic default-output API:
            # generic callers may retain results across later collectives.
            assert self._pcie_dma is not None
            if runtime_stream is not None:
                with torch.cuda.stream(runtime_stream):
                    self._pcie_dma.all_reduce(inp, out=inp)
                    ops.fused_add_rms_norm(
                        inp, residual, weight, epsilon
                    )
            else:
                self._pcie_dma.all_reduce(inp, out=inp)
                ops.fused_add_rms_norm(inp, residual, weight, epsilon)
            return True

        assert self._pcie_runtime is not None
        self._pcie_runtime.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=inp,
            residual_out=residual,
            stream=runtime_stream,
        )
        return True
'''


def main() -> int:
    target = find_target()
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {target}")
        return 0
    if text.count(OLD) != 1:
        raise RuntimeError(
            f"expected exactly one v20-r9 fused method in {target}, "
            f"found {text.count(OLD)}"
        )
    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    patched = target.read_text(encoding="utf-8")
    if patched.count(MARKER) != 1 or "allreduce_out = torch.empty_like" in patched:
        raise RuntimeError("post-patch invariant failed")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
