#!/usr/bin/env python3
"""Smoke the guarded SwinIR attention patch without requiring an MPS device."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


EXTENSION_ROOT = Path(os.environ["FORGE_APPLE_TEST_EXTENSION_ROOT"]).resolve()
sys.path.insert(0, str(EXTENSION_ROOT))

from spandrel.architectures.SwinIR.__arch.SwinIR import WindowAttention


def main() -> int:
    torch.manual_seed(12345)
    module = WindowAttention(dim=240, window_size=(8, 8), num_heads=8).eval()
    fp32_input = torch.randn((2, 64, 240), dtype=torch.float32)
    fp32_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    fp32_reference = module(fp32_input, mask=fp32_mask)

    from forge_apple_accelerator.runtime import swinir_bf16

    patch_status = swinir_bf16.install()
    assert patch_status["compatible"], patch_status
    fp32_candidate = module(fp32_input, mask=fp32_mask)
    assert torch.equal(fp32_reference, fp32_candidate), "FP32 fallback changed"

    module.bfloat16()
    bf16_output = module(fp32_input.bfloat16(), mask=fp32_mask)
    assert bf16_output.dtype is torch.bfloat16
    assert torch.isfinite(bf16_output).all()
    print(
        json.dumps(
            {
                "ok": True,
                "fp32_byte_exact": True,
                "bf16_finite": True,
                "bf16_shape": list(bf16_output.shape),
                "patch": patch_status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
