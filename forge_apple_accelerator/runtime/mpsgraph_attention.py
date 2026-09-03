import os
from pathlib import Path
from typing import Optional

import torch

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, EXTENSION_ROOT, import_bridge


_FALSE_VALUES = {"", "0", "false", "no", "off"}
_MODULE = None
_LOAD_ATTEMPTED = False
_LAST_ERROR = ""
_HYBRID_BACKEND = "torch-sdpa-self-mpsgraph-cross"
_SUPPORTED_BACKENDS = {"mpsgraph-sdpa", "mpsgraph-softmax-api", "mpsgraph-manual-softmax", _HYBRID_BACKEND}
_SUPPORTED_TOKEN_COUNTS = {2304, 3456, 4096, 5184, 6144, 9216}
_SUPPORTED_CROSS_TOKEN_COUNTS = {4096, 9216}
_SUPPORTED_CROSS_KV_TOKEN_COUNTS = {512}
_LAST_ROUTE: dict[str, object] = {"status": "idle", "reason": "", "backend": "off"}


def _enabled() -> bool:
    return os.getenv("FORGE_APPLE_ATTENTION_BACKEND", "off").strip().lower() in _SUPPORTED_BACKENDS


def _backend() -> str:
    return os.getenv("FORGE_APPLE_ATTENTION_BACKEND", "off").strip().lower()


def _cross_enabled() -> bool:
    return os.getenv("FORGE_APPLE_ATTENTION_CROSS", "off").strip().lower() not in _FALSE_VALUES


def last_error() -> str:
    return _LAST_ERROR


def last_route() -> dict[str, object]:
    return dict(_LAST_ROUTE)


def _set_route(status: str, reason: str = "", **fields):
    global _LAST_ROUTE
    _LAST_ROUTE = {
        "status": status,
        "reason": reason,
        "backend": _backend(),
        **fields,
    }


def _load_extension():
    global _MODULE, _LOAD_ATTEMPTED, _LAST_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True

    manifest_path = Path(os.getenv("FORGE_APPLE_MPSGRAPH_ATTENTION_MANIFEST", str(DEFAULT_MANIFEST_PATH))).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = EXTENSION_ROOT / manifest_path
    try:
        module, _manifest = import_bridge(
            manifest_path,
            required_attributes=("mpsgraph_attention",),
            torch_module=torch,
        )
        _MODULE = module
        _LAST_ERROR = ""
        return _MODULE
    except Exception as exc:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _flatten_sequence_dims(tensor: torch.Tensor) -> torch.Tensor:
    shape = _shape(tensor)
    if len(shape) <= 4:
        return tensor
    batch = shape[0]
    heads = shape[-2]
    head_dim = shape[-1]
    tokens = 1
    for dim in shape[1:-2]:
        tokens *= dim
    return tensor.reshape(batch, tokens, heads, head_dim)


def _reject(reason: str, q: torch.Tensor | None = None):
    fields = {}
    if q is not None:
        fields["shape"] = _shape(q)
        fields["dtype"] = str(q.dtype)
        fields["device"] = q.device.type
        fields["is_contiguous"] = bool(q.is_contiguous())
    _set_route("fallback", reason, **fields)
    return None


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_self_attention: bool) -> Optional[torch.Tensor]:
    global _LAST_ERROR
    if not _enabled():
        return _reject("backend_off", q)
    q = _flatten_sequence_dims(q)
    k = _flatten_sequence_dims(k)
    v = _flatten_sequence_dims(v)
    q_shape = _shape(q)
    k_shape = _shape(k)
    v_shape = _shape(v)
    if len(q_shape) != 4:
        return _reject("rank_not_4", q)
    batch, tokens, heads, head_dim = q_shape
    if len(k_shape) != 4 or len(v_shape) != 4:
        return _reject("kv_rank_not_4", q)
    if k_shape != v_shape:
        return _reject("kv_shape_mismatch", q)
    if batch != 2 or heads != 16 or head_dim != 128 or k_shape[0] != batch or k_shape[2] != heads or k_shape[3] != head_dim:
        return _reject("unsupported_bhd", q)
    kv_tokens = k_shape[1]
    # On the qualified nightly, native Torch SDPA wins at the normal 4096-token
    # Anima shape.  The existing MPSGraph route remains the qualified choice at
    # the 9216-token Hi-Res shape, so hybrid mode only declines the former.
    if is_self_attention and _backend() == _HYBRID_BACKEND and tokens == 4096:
        return _reject("hybrid_self_routes_to_torch_sdpa", q)
    if is_self_attention:
        if q_shape != k_shape:
            return _reject("self_shape_mismatch", q)
        if tokens not in _SUPPORTED_TOKEN_COUNTS:
            return _reject("unsupported_token_count", q)
    else:
        if not _cross_enabled():
            return _reject("cross_attention_off", q)
        if _backend() not in {"mpsgraph-sdpa", _HYBRID_BACKEND}:
            return _reject("cross_requires_sdpa", q)
        if tokens not in _SUPPORTED_CROSS_TOKEN_COUNTS:
            return _reject("unsupported_cross_token_count", q)
        if kv_tokens not in _SUPPORTED_CROSS_KV_TOKEN_COUNTS:
            return _reject("unsupported_cross_kv_token_count", q)
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return _reject("unsupported_dtype", q)
    if q.device.type != "mps" or k.device.type != "mps" or v.device.type != "mps":
        return _reject("unsupported_device", q)
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return _reject("non_contiguous", q)

    backend = _backend()
    if backend not in _SUPPORTED_BACKENDS:
        return _reject("unsupported_backend", q)
    module = _load_extension()
    if module is None:
        return _reject(f"extension_unavailable:{_LAST_ERROR}", q)
    try:
        native_backend = "mpsgraph-sdpa" if backend == _HYBRID_BACKEND else backend
        output = module.mpsgraph_attention(q, k, v, native_backend)
    except Exception as exc:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return _reject(f"runtime_error:{_LAST_ERROR}", q)
    expected_output_shape = (batch, tokens, heads * head_dim)
    if output is None or _shape(output) != expected_output_shape or output.dtype != torch.bfloat16 or output.device.type != "mps":
        return _reject("bad_output", q)
    _set_route("hit", "", shape=q_shape, kv_shape=k_shape, output_shape=expected_output_shape, token_count=tokens, kv_token_count=kv_tokens)
    return output
