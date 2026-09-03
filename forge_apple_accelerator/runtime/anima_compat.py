"""PyTorch 2.12-exact Anima primitives for newer, faster MPS runtimes.

These routes exist solely to preserve the stable Forge denoiser trajectory when
running a newer PyTorch build. Unsupported tensors fail closed so Forge's normal
operators remain the fallback.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, EXTENSION_ROOT, import_bridge
from forge_apple_accelerator.runtime import metal_kernels


_SUPPORTED_BACKENDS = {"torch212-exact"}
_LOCK = threading.Lock()
_EXTENSION_MODULE = None
_EXTENSION_ERROR: str | None = None
_LAST_ROUTE: dict[str, Any] = {}
_STATS = {
    "layer_norm_calls": 0,
    "layer_norm_routed": 0,
    "rms_norm_calls": 0,
    "rms_norm_routed": 0,
    "gelu_calls": 0,
    "gelu_routed": 0,
    "normal_calls": 0,
    "normal_routed": 0,
    "fallbacks": 0,
    "load_errors": 0,
    "last_error": "",
}


def _backend() -> str:
    return os.getenv("FORGE_APPLE_ANIMA_COMPAT_BACKEND", "off").strip().lower()


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
    try:
        module, _manifest = import_bridge(
            _extension_manifest_path(),
            required_attributes=("mpsgraph_legacy_gelu", "mpsgraph_legacy_normal"),
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


def _record(operation: str, *, hit: bool, reason: str = "") -> None:
    global _LAST_ROUTE
    with _LOCK:
        if hit:
            _STATS[f"{operation}_routed"] += 1
            _STATS["last_error"] = ""
        else:
            _STATS["fallbacks"] += 1
            _STATS["last_error"] = reason
        _LAST_ROUTE = {
            "backend": _backend(),
            "primitive": operation,
            "status": "hit" if hit else "fallback",
            "reason": reason,
        }


def _eligible(input_tensor: torch.Tensor, *, features: int) -> str:
    if _backend() not in _SUPPORTED_BACKENDS:
        return "disabled"
    if torch.jit.is_tracing():
        return "tracing"
    if input_tensor.device.type != "mps":
        return "device"
    if input_tensor.dtype != torch.bfloat16:
        return "dtype"
    if input_tensor.dim() == 0 or input_tensor.shape[-1] != features:
        return "features"
    if not input_tensor.is_contiguous() or input_tensor.storage_offset() != 0:
        return "layout"
    return ""


def layer_norm(
    input_tensor: torch.Tensor,
    *,
    normalized_shape: tuple[int, ...],
    eps: float,
    elementwise_affine: bool,
) -> torch.Tensor | None:
    with _LOCK:
        _STATS["layer_norm_calls"] += 1
    reason = _eligible(input_tensor, features=2048)
    if reason:
        _record("layer_norm", hit=False, reason=reason)
        return None
    if tuple(normalized_shape) != (2048,) or elementwise_affine or float(eps) != 1.0e-6:
        _record("layer_norm", hit=False, reason="semantics")
        return None
    try:
        output = metal_kernels.legacy_layer_norm(input_tensor, eps=eps, allow_experimental=True)
    except Exception as exc:
        _record("layer_norm", hit=False, reason=f"{type(exc).__name__}: {exc}")
        return None
    if output is None:
        _record("layer_norm", hit=False, reason="kernel_declined")
        return None
    _record("layer_norm", hit=True)
    return output


def rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor | None:
    with _LOCK:
        _STATS["rms_norm_calls"] += 1
    if tuple(normalized_shape) not in {(128,), (2048,)}:
        _record("rms_norm", hit=False, reason="normalized_shape")
        return None
    features = int(normalized_shape[0])
    reason = _eligible(input_tensor, features=features)
    if reason:
        _record("rms_norm", hit=False, reason=reason)
        return None
    if weight is None or tuple(weight.shape) != (features,) or float(eps) != 1.0e-6:
        _record("rms_norm", hit=False, reason="semantics")
        return None
    if (
        weight.device != input_tensor.device
        or weight.dtype != input_tensor.dtype
        or not weight.is_contiguous()
        or weight.storage_offset() != 0
    ):
        _record("rms_norm", hit=False, reason="weight")
        return None
    try:
        output = metal_kernels.legacy_rms_norm(input_tensor, weight, eps=eps, allow_experimental=True)
    except Exception as exc:
        _record("rms_norm", hit=False, reason=f"{type(exc).__name__}: {exc}")
        return None
    if output is None:
        _record("rms_norm", hit=False, reason="kernel_declined")
        return None
    _record("rms_norm", hit=True)
    return output


def gelu(input_tensor: torch.Tensor, *, approximate: str) -> torch.Tensor | None:
    with _LOCK:
        _STATS["gelu_calls"] += 1
    reason = _eligible(input_tensor, features=8192)
    if reason:
        _record("gelu", hit=False, reason=reason)
        return None
    if approximate != "none":
        _record("gelu", hit=False, reason="approximation")
        return None
    module = _load_extension()
    if module is None:
        _record("gelu", hit=False, reason=_EXTENSION_ERROR or "extension_unavailable")
        return None
    try:
        output = module.mpsgraph_legacy_gelu(input_tensor)
    except Exception as exc:
        _record("gelu", hit=False, reason=f"{type(exc).__name__}: {exc}")
        return None
    if output.dtype != input_tensor.dtype or output.device.type != "mps" or output.shape != input_tensor.shape:
        _record("gelu", hit=False, reason="extension_output")
        return None
    _record("gelu", hit=True)
    return output


def legacy_randn(shape: tuple[int, ...], *, generator=None) -> torch.Tensor | None:
    with _LOCK:
        _STATS["normal_calls"] += 1
    if _backend() not in _SUPPORTED_BACKENDS:
        _record("normal", hit=False, reason="disabled")
        return None
    if not torch.backends.mps.is_available():
        _record("normal", hit=False, reason="mps_unavailable")
        return None
    try:
        normalized_shape = tuple(int(dimension) for dimension in shape)
    except Exception as exc:
        _record("normal", hit=False, reason=f"shape:{type(exc).__name__}: {exc}")
        return None
    if not normalized_shape or any(dimension < 0 for dimension in normalized_shape):
        _record("normal", hit=False, reason="shape")
        return None
    module = _load_extension()
    if module is None:
        _record("normal", hit=False, reason=_EXTENSION_ERROR or "extension_unavailable")
        return None
    try:
        output = module.mpsgraph_legacy_normal(normalized_shape, generator)
    except Exception as exc:
        _record("normal", hit=False, reason=f"{type(exc).__name__}: {exc}")
        return None
    if output.device.type != "mps" or output.dtype != torch.float32 or tuple(output.shape) != normalized_shape:
        _record("normal", hit=False, reason="extension_output")
        return None
    _record("normal", hit=True)
    return output


def last_route() -> dict[str, Any]:
    with _LOCK:
        return dict(_LAST_ROUTE)


def stats(reset: bool = False) -> dict[str, Any]:
    with _LOCK:
        payload = {
            **_STATS,
            "backend": _backend(),
            "extension_loaded": _EXTENSION_MODULE is not None,
            "extension_error": _EXTENSION_ERROR,
            "time_unix": time.time(),
        }
        if reset:
            _STATS.update(
                {
                    "layer_norm_calls": 0,
                    "layer_norm_routed": 0,
                    "rms_norm_calls": 0,
                    "rms_norm_routed": 0,
                    "gelu_calls": 0,
                    "gelu_routed": 0,
                    "normal_calls": 0,
                    "normal_routed": 0,
                    "fallbacks": 0,
                    "load_errors": 0,
                    "last_error": "",
                }
            )
    return payload
