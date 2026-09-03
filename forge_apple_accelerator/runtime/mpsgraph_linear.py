from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, EXTENSION_ROOT, import_bridge


_FALSE_VALUES = {"", "0", "false", "no", "off"}
_SUPPORTED_BACKENDS = {"mpsgraph-linear"}
_LOCK = threading.Lock()
_EXTENSION_MODULE = None
_EXTENSION_ERROR: str | None = None
_STATS = {
    "calls": 0,
    "routed": 0,
    "fallbacks": 0,
    "load_errors": 0,
    "last_error": "",
}


def _extension_manifest_path() -> Path:
    configured = os.getenv("FORGE_APPLE_MPSGRAPH_EXTENSION_MANIFEST", "").strip()
    if configured:
        path = Path(configured).expanduser()
    else:
        path = DEFAULT_MANIFEST_PATH
    return path if path.is_absolute() else EXTENSION_ROOT / path


def _load_extension():
    global _EXTENSION_MODULE, _EXTENSION_ERROR
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE
    if _EXTENSION_ERROR is not None:
        return None

    manifest_path = _extension_manifest_path()
    try:
        module, _manifest = import_bridge(
            manifest_path,
            required_attributes=("mpsgraph_linear",),
            torch_module=torch,
        )
        _EXTENSION_MODULE = module
        return module
    except Exception as exc:
        _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
        with _LOCK:
            _STATS["load_errors"] += 1
            _STATS["last_error"] = _EXTENSION_ERROR
        return None


def _enabled_backend() -> str:
    backend = os.getenv("FORGE_APPLE_LINEAR_BACKEND", "off").strip().lower()
    if backend in _FALSE_VALUES:
        return "off"
    return backend


def _is_mps(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "mps"


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _fallback(reason: str):
    with _LOCK:
        _STATS["fallbacks"] += 1
        _STATS["last_error"] = reason
    return None


def linear(input_tensor: torch.Tensor, weight: torch.Tensor, *, kind: str = "") -> torch.Tensor | None:
    """Run a BF16 linear op through the audited MPSGraph extension when explicitly enabled.

    The route is intentionally conservative: no bias, MPS BF16 tensors only, 2D
    flattened matmul through the extension, then reshape back to PyTorch's
    `nn.Linear` output shape. Unsupported cases return None so callers can fall
    back to the existing PyTorch/MPS path.
    """

    backend = _enabled_backend()
    with _LOCK:
        _STATS["calls"] += 1
    if backend == "off":
        return _fallback("disabled")
    if backend not in _SUPPORTED_BACKENDS:
        return _fallback(f"unsupported backend {backend}")
    if torch.jit.is_tracing():
        return _fallback("tracing")
    if input_tensor.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return _fallback("dtype")
    if not (_is_mps(input_tensor) and _is_mps(weight)):
        return _fallback("device")
    if weight.dim() != 2 or input_tensor.dim() < 2:
        return _fallback("rank")
    if input_tensor.shape[-1] != weight.shape[1]:
        return _fallback("features")
    if not weight.is_contiguous():
        return _fallback("weight_layout")
    if not input_tensor.is_contiguous():
        return _fallback("input_layout")

    module = _load_extension()
    if module is None:
        return _fallback(_EXTENSION_ERROR or "extension unavailable")

    original_shape = tuple(input_tensor.shape[:-1]) + (int(weight.shape[0]),)
    try:
        flattened = input_tensor.reshape(-1, input_tensor.shape[-1])
        if not flattened.is_contiguous():
            return _fallback("flattened_layout")
        output = module.mpsgraph_linear(flattened, weight)
        if output.dtype != input_tensor.dtype or output.device.type != "mps":
            return _fallback("extension_output")
        result = output.reshape(original_shape)
        if not torch.isfinite(result).all().item():
            return _fallback("nonfinite")
        with _LOCK:
            _STATS["routed"] += 1
            _STATS["last_error"] = ""
        return result
    except Exception as exc:
        return _fallback(f"{type(exc).__name__}: {exc}")


def stats(reset: bool = False) -> dict[str, Any]:
    with _LOCK:
        payload = dict(_STATS)
        payload["extension_loaded"] = _EXTENSION_MODULE is not None
        payload["extension_error"] = _EXTENSION_ERROR
        payload["backend"] = _enabled_backend()
        payload["time_unix"] = time.time()
        if reset:
            _STATS.update({"calls": 0, "routed": 0, "fallbacks": 0, "load_errors": 0, "last_error": ""})
    return payload


def describe_candidate(input_tensor: torch.Tensor, weight: torch.Tensor, *, kind: str = "") -> dict[str, Any]:
    return {
        "kind": kind,
        "backend": _enabled_backend(),
        "input": _tensor_meta(input_tensor),
        "weight": _tensor_meta(weight),
        "supported": (
            _enabled_backend() in _SUPPORTED_BACKENDS
            and input_tensor.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and _is_mps(input_tensor)
            and _is_mps(weight)
            and weight.dim() == 2
            and input_tensor.dim() >= 2
            and input_tensor.shape[-1] == weight.shape[1]
            and input_tensor.is_contiguous()
            and weight.is_contiguous()
        ),
    }
