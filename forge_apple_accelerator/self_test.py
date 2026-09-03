#!/usr/bin/env python3
"""Run bounded parity, routing, and memory checks for the Apple provider."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, validate_manifest


def _timed(torch, function: Callable[[], Any]) -> tuple[Any, float]:
    torch.mps.synchronize()
    started = time.perf_counter()
    value = function()
    torch.mps.synchronize()
    return value, time.perf_counter() - started


def _tensor_error(torch, actual, expected) -> dict[str, Any]:
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    difference = (actual_cpu - expected_cpu).abs()
    denominator = expected_cpu.abs().mean().item() + 1e-12
    return {
        "exact_equal": bool(torch.equal(actual.detach().cpu(), expected.detach().cpu())),
        "finite": bool(torch.isfinite(actual_cpu).all().item()),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rel_mean_abs": float(difference.mean().item() / denominator),
    }


def _current_allocated(torch) -> int:
    function = getattr(torch.mps, "current_allocated_memory", None)
    return int(function()) if function is not None else 0


def run(*, rope_capture: Path | None = None, attention_capture: Path | None = None) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "torch": torch.__version__,
        },
        "checks": {},
    }
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        report.update({"ok": False, "reason": "unsupported_host"})
        return report
    if not torch.backends.mps.is_available():
        report.update({"ok": False, "reason": "mps_unavailable"})
        return report

    valid, reason, manifest, _module_path = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
    report["checks"]["manifest"] = {
        "passed": valid,
        "reason": reason,
        "schema_version": manifest.get("schema_version"),
    }
    if not valid:
        report.update({"ok": False, "reason": reason})
        return report

    from forge_apple_accelerator.runtime import metal_kernels, mpsgraph_attention

    os.environ["FORGE_APPLE_METAL_KERNELS"] = "rope"
    torch.manual_seed(1729)
    if rope_capture is not None:
        payload = torch.load(rope_capture, map_location="cpu", weights_only=False)
        inputs = payload["tensors"]["inputs"]
        outputs = payload["tensors"]["outputs"]
        xq = inputs["xq"].to("mps")
        xk = inputs["xk"].to("mps")
        freqs = inputs["freqs"].to("mps")
        expected_q = outputs["q"].to("mps")
        expected_k = outputs["k"].to("mps")
        rope_shape = tuple(xq.shape)
        reference_name = f"capture:{rope_capture}"
    else:
        rope_shape = (2, 256, 16, 128)
        xq = torch.randn(rope_shape, dtype=torch.float32).to(dtype=torch.bfloat16, device="mps")
        xk = torch.randn(rope_shape, dtype=torch.float32).to(dtype=torch.bfloat16, device="mps")
        freqs = torch.randn((rope_shape[1], rope_shape[-1] // 2, 2, 2), dtype=torch.float32, device="mps")
        half = rope_shape[-1] // 2

        def reference(tensor):
            first = tensor[..., :half].to(freqs.dtype)
            second = tensor[..., half:].to(freqs.dtype)
            cos = freqs[..., 0, 0].unsqueeze(0).unsqueeze(2)
            neg_sin = freqs[..., 0, 1].unsqueeze(0).unsqueeze(2)
            sin = freqs[..., 1, 0].unsqueeze(0).unsqueeze(2)
            cos2 = freqs[..., 1, 1].unsqueeze(0).unsqueeze(2)
            return torch.cat([cos * first + neg_sin * second, sin * first + cos2 * second], dim=-1).type_as(tensor)

        expected_q, expected_k = reference(xq), reference(xk)
        reference_name = "torch_reference"
    rope_result, rope_seconds = _timed(torch, lambda: metal_kernels.rope_pair(xq, xk, freqs))
    if rope_result is None:
        rope_check = {"passed": False, "reason": "provider_declined"}
    else:
        actual_q, actual_k = rope_result
        q_error = _tensor_error(torch, actual_q, expected_q)
        k_error = _tensor_error(torch, actual_k, expected_k)
        rope_check = {
            "passed": q_error["exact_equal"] and k_error["exact_equal"],
            "reference": reference_name,
            "seconds": rope_seconds,
            "q_error": q_error,
            "k_error": k_error,
        }
    report["checks"]["metal_rope_exact"] = rope_check

    xq = xk = freqs = expected_q = expected_k = rope_result = None
    gc.collect()

    os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "mpsgraph-sdpa"
    os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "1"
    if attention_capture is not None:
        payload = torch.load(attention_capture, map_location="cpu", weights_only=False)
        inputs = payload["tensors"]["inputs"]
        outputs = payload["tensors"]["outputs"]
        q = inputs["q"].to("mps")
        k = inputs["k"].to("mps")
        v = inputs["v"].to("mps")
        expected_attention = outputs["output"].to("mps")
        attention_shape = tuple(q.shape)
        reference_seconds = None
        attention_reference = f"capture:{attention_capture}"
    else:
        attention_shape = (2, 2304, 16, 128)
        q = torch.randn(attention_shape, dtype=torch.float32).to(dtype=torch.bfloat16, device="mps")
        k = torch.randn(attention_shape, dtype=torch.float32).to(dtype=torch.bfloat16, device="mps")
        v = torch.randn(attention_shape, dtype=torch.float32).to(dtype=torch.bfloat16, device="mps")
        expected_attention = None
        reference_seconds = None
        attention_reference = "torch_sdpa"

    def torch_attention():
        output = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return output.transpose(1, 2).reshape(attention_shape[0], attention_shape[1], -1)

    if expected_attention is None:
        expected_attention, reference_seconds = _timed(torch, torch_attention)
    candidate, candidate_seconds = _timed(
        torch,
        lambda: mpsgraph_attention.attention_core(q, k, v, is_self_attention=True),
    )
    attention_route = mpsgraph_attention.last_route()
    if candidate is None:
        attention_check = {"passed": False, "route": attention_route}
    else:
        attention_error = _tensor_error(torch, candidate, expected_attention)
        attention_check = {
            "passed": (
                attention_error["finite"]
                and attention_error["rel_mean_abs"] <= 0.001
                and attention_error["max_abs"] <= 0.0078125
                and attention_route.get("status") == "hit"
            ),
            "reference": attention_reference,
            "reference_seconds": reference_seconds,
            "candidate_seconds": candidate_seconds,
            "speedup": reference_seconds / candidate_seconds if reference_seconds is not None and candidate_seconds else None,
            "error": attention_error,
            "route": attention_route,
        }
    report["checks"]["mpsgraph_attention_parity"] = attention_check

    _warm, _warm_seconds = _timed(torch, lambda: mpsgraph_attention.attention_core(q, k, v, is_self_attention=True))
    del _warm
    torch.mps.synchronize()
    memory_before = _current_allocated(torch)
    repeated = None
    for _index in range(3):
        repeated, _elapsed = _timed(torch, lambda: mpsgraph_attention.attention_core(q, k, v, is_self_attention=True))
        del repeated
    torch.mps.synchronize()
    memory_after = _current_allocated(torch)
    memory_growth = max(0, memory_after - memory_before)
    memory_limit = 64 * 1024 * 1024
    report["checks"]["repeated_call_memory"] = {
        "passed": memory_growth <= memory_limit,
        "before_bytes": memory_before,
        "after_bytes": memory_after,
        "growth_bytes": memory_growth,
        "limit_bytes": memory_limit,
    }

    unsupported = mpsgraph_attention.attention_core(q[:, :64], k[:, :64], v[:, :64], is_self_attention=True)
    unsupported_route = mpsgraph_attention.last_route()
    report["checks"]["unsupported_shape_fallback"] = {
        "passed": unsupported is None and unsupported_route.get("status") == "fallback",
        "route": unsupported_route,
    }

    report["ok"] = all(bool(check.get("passed")) for check in report["checks"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--rope-capture", type=Path)
    parser.add_argument("--attention-capture", type=Path)
    args = parser.parse_args(argv)
    report = run(rope_capture=args.rope_capture, attention_capture=args.attention_capture)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.json_out)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
