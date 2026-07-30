"""CPU and source gates for the EXL3 mixed-K persistent output buffer."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "tr3-325-public" / "patch_exl3_mixk.py"


def test_persistent_cast_is_bit_exact_and_reuses_storage() -> None:
    torch.manual_seed(20260729)
    accum = torch.randn((3072, 6144), dtype=torch.float32)
    expected = accum.to(torch.bfloat16)
    output = torch.empty_like(accum, dtype=torch.bfloat16)
    pointer = output.data_ptr()

    output.copy_(accum)

    assert output.data_ptr() == pointer
    assert torch.equal(output, expected)

    accum.mul_(0.5)
    output.copy_(accum)
    assert output.data_ptr() == pointer
    assert torch.equal(output, accum.to(torch.bfloat16))


def test_patch_fails_closed_on_v4_contract() -> None:
    source = PATCH.read_text()
    assert "EXL3-MIXK-PATCH v4 (persistent output buffer)" in source
    assert '"output": output' in source
    assert 'output = runtime["output"][:m]' in source
    assert "output.copy_(accum)" in source
    assert "return output" in source
    assert "return accum.to(x.dtype)" not in source
