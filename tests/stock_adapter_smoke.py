#!/usr/bin/env python3
"""Smoke the extension preload against a pristine stock Forge Neo checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> int:
    extension_root = Path(os.environ["FORGE_APPLE_TEST_EXTENSION_ROOT"]).resolve()
    stock_root = Path.cwd().resolve()
    sys.path.insert(0, str(extension_root))
    sys.path.insert(0, str(stock_root))
    sys.path.insert(0, str(stock_root / "modules_forge/packages"))
    # The smoke intentionally exercises CPU fallbacks and can run inside a
    # sandbox where MPS device discovery is unavailable.
    if "--cpu" not in sys.argv:
        sys.argv.append("--cpu")

    exact = os.getenv("FORGE_APPLE_TEST_EXACT", "0") == "1"
    os.environ["FORGE_APPLE_ACCELERATOR_MODE"] = "inherit" if exact else "visual-fast"
    if exact:
        os.environ["FORGE_APPLE_ANIMA_COMPAT_BACKEND"] = "torch212-exact"
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "off"

    import torch
    from backend.nn import anima

    torch.manual_seed(12345)
    attention = anima.SelfCrossAttention(query_dim=48, context_dim=None, n_heads=4, head_dim=12)
    x = torch.randn((1, 4, 48), dtype=torch.float32)
    video = torch.zeros((1, 1, 2, 2, 48), dtype=torch.float32)
    rope = anima.VideoRopePosition3DEmb(12, 1.0, 1.0, 1.0)(video, torch.device("cpu")).unsqueeze(1).unsqueeze(0)
    q_before, k_before, v_before = attention.compute_qkv(x, rope_emb=rope)
    output_before = attention.compute_attention(q_before, k_before, v_before)

    preload_path = extension_root / "preload.py"
    spec = importlib.util.spec_from_file_location("forge_apple_accelerator_test_preload", preload_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {preload_path}")
    preload_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preload_module)
    preload_module.preload(argparse.ArgumentParser(add_help=False))

    q_after, k_after, v_after = attention.compute_qkv(x, rope_emb=rope)
    output_after = attention.compute_attention(q_after, k_after, v_after)
    assert torch.equal(q_before, q_after), "q fallback changed"
    assert torch.equal(k_before, k_after), "k fallback changed"
    assert torch.equal(v_before, v_after), "v fallback changed"
    assert torch.equal(output_before, output_after), "attention fallback changed"

    from modules import shared_init

    shared_init.initialize()
    script_path = extension_root / "scripts/forge_apple_accelerator.py"
    script_spec = importlib.util.spec_from_file_location("forge_apple_accelerator_test_script", script_path)
    if script_spec is None or script_spec.loader is None:
        raise RuntimeError(f"cannot load {script_path}")
    script_module = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script_module)

    from forge_apple_accelerator import stock_adapter

    adapter = stock_adapter.status()
    assert adapter["installed"] is True, adapter
    assert adapter["compatible"] is True, adapter
    assert adapter["upscaler_settings_applied"] is True, adapter
    if exact:
        assert "GPT2FeedForward.forward" in adapter["patched"], adapter
        assert adapter["rng_patched"] is True, adapter
    print(
        json.dumps(
            {
                "ok": True,
                "stock_root": str(stock_root),
                "exact": exact,
                "adapter": adapter,
                "fallback_byte_exact": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
