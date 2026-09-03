"""Experimental whole-Anima MLX denoiser provider.

The route has one device-resident boundary in each direction per invocation:
all prepared Forge inputs are imported with DLPack ``copy=False`` and one MLX
output is returned through ``torch.from_dlpack``.  The implementation is
deliberately narrow and fail-open; unsupported state returns to the literal
Forge diffusion-model call in ``KModel.apply_model``.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib
import json
import os
import sys
import threading
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from . import mlx_sync_bridge


_BACKEND = "mlx-anima-experimental"
_REQUIRED_TORCH = "2.12.1"
_REQUIRED_MLX = "0.32.2"
_FORGE_ROOT = Path(__file__).resolve().parents[4]
_ANIMA_MLX_SOURCE = _FORGE_ROOT / "scripts/apple_mlx/anima_mlx.py"
_DEFAULT_MAPPING_REPORT = _FORGE_ROOT / "models/apple_mlx/phase2-denoiser/weight_mapping_manifest.json"
_SUPPORTED_TRANSFORMER_PATCH_KEYS = (
    "patches",
    "patches_replace",
    "block_modifiers",
    "block_inner_modifiers",
    "conditioning_modifiers",
    "model_function_wrapper",
    "controlnet_conditioning_modifiers",
    "controlnet_model_function_wrapper",
)


class _Declined(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclasses.dataclass
class _ResidentState:
    fingerprint: str
    fingerprint_payload: dict[str, Any]
    checkpoint: Path
    checkpoint_sha256: str
    weights: Any
    compiled: Any
    weight_arrays: list[Any]
    weight_owners: list[Any]
    weight_bytes: int
    copied_weight_bytes: int
    shared_weight_bytes: int
    weight_residency: str
    load_seconds: float
    graph_construction_seconds: float
    mapping_report: dict[str, Any]
    invocation_count: int = 0
    compiled_signatures: set[str] = dataclasses.field(default_factory=set)


_LOCK = threading.RLock()
_STATE: _ResidentState | None = None
_REJECTED: dict[str, str] = {}
_LAST_ROUTE: dict[str, Any] = {"status": "fallback", "reason": "not_called", "backend": _BACKEND}
_CHECKPOINT_HASHES: dict[tuple[str, int, int], str] = {}
_STATS: dict[str, Any] = {
    "calls": 0,
    "hits": 0,
    "fallbacks": Counter(),
    "compile_cache_hits": 0,
    "compile_cache_misses": 0,
    "state_loads": 0,
    "state_invalidations": 0,
    "cold_compile_seconds": None,
    "last_warm_execution_seconds": None,
    "last_total_seconds": None,
    "zero_copy_input_activations": 0,
    "zero_copy_output_activations": 0,
    "copied_activation_count": 0,
    "resident_weight_bytes": 0,
    "copied_weight_bytes": 0,
    "shared_weight_bytes": 0,
    "diagnostic_finite_checks": 0,
    "gil_safe_native_sync_calls": 0,
    "last_error": "",
}


def _diagnostic_finite_check_enabled() -> bool:
    return os.getenv("FORGE_APPLE_MLX_VALIDATE_FINITE", "").strip().lower() in {"1", "true", "yes", "on"}


def _lifetime_diagnostics_enabled() -> bool:
    return os.getenv("FORGE_APPLE_MLX_LIFETIME_DIAGNOSTICS", "").strip().lower() in {"1", "true", "yes", "on"}


def _synchronization_description() -> str:
    backend = mlx_sync_bridge.selected_backend()
    if backend == mlx_sync_bridge.BACKEND_GIL_SAFE_NATIVE:
        return "Torch fence before MLX; MLX eval + exact native GIL-released fence before Torch"
    return "Torch fence before MLX; Python MLX eval+fence before Torch"


def _synchronize(mx: Any) -> None:
    try:
        mlx_sync_bridge.synchronize(mx)
        if mlx_sync_bridge.selected_backend() == mlx_sync_bridge.BACKEND_GIL_SAFE_NATIVE:
            _STATS["gil_safe_native_sync_calls"] += 1
    except mlx_sync_bridge.BridgeUnavailable as exc:
        raise _Declined(f"gil_safe_sync_unavailable:{exc}") from exc


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _CHECKPOINT_HASHES.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _CHECKPOINT_HASHES[key] = value
    return value


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _mlx_site() -> str:
    return os.getenv("FORGE_APPLE_MLX_SITE", "").strip()


def _weight_residency() -> str:
    return os.getenv("FORGE_APPLE_MLX_WEIGHT_RESIDENCY", "shared-forge").strip().lower()


def _load_runtime() -> tuple[Any, Any, Any]:
    site = _mlx_site()
    if site:
        resolved = str(Path(site).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    import torch
    import mlx.core as mx

    if str(torch.__version__) != _REQUIRED_TORCH:
        raise _Declined(f"unsupported_torch_version:{torch.__version__}")
    try:
        mlx_version = metadata.version("mlx")
    except metadata.PackageNotFoundError:
        mlx_version = "unknown"
    if mlx_version != _REQUIRED_MLX or not hasattr(mx, "from_dlpack"):
        raise _Declined(f"unsupported_mlx_version:{mlx_version}")
    if not torch.backends.mps.is_available() or not mx.metal.is_available():
        raise _Declined("metal_unavailable")

    scripts = str(_ANIMA_MLX_SOURCE.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    anima_mlx = importlib.import_module("anima_mlx")
    if Path(anima_mlx.__file__).resolve() != _ANIMA_MLX_SOURCE.resolve():
        raise _Declined("unexpected_anima_mlx_source")
    return torch, mx, anima_mlx


def _is_anima(model: Any) -> bool:
    cls = model.__class__
    return cls.__name__ == "Anima" and cls.__module__ == "backend.nn.anima" and len(getattr(model, "blocks", ())) == 28


def _transformer_patch_reason(options: Any) -> str | None:
    if not isinstance(options, dict):
        return "non_dict_transformer_options"
    active = [key for key in _SUPPORTED_TRANSFORMER_PATCH_KEYS if options.get(key)]
    return f"unsupported_transformer_patches:{','.join(active)}" if active else None


def _tensor_reason(name: str, tensor: Any, expected_shape: tuple[int, ...], expected_dtype: Any) -> str | None:
    if not hasattr(tensor, "device") or tensor.device.type != "mps":
        return f"{name}_not_mps"
    if tuple(tensor.shape) != expected_shape:
        return f"unsupported_{name}_shape:{tuple(tensor.shape)}"
    if tensor.dtype is not expected_dtype:
        return f"unsupported_{name}_dtype:{tensor.dtype}"
    if not tensor.is_contiguous():
        return f"unsupported_{name}_layout:noncontiguous"
    if tensor.storage_offset() != 0:
        return f"unsupported_{name}_layout:storage_offset"
    return None


def _model_fingerprint(owner: Any, model: Any, checkpoint: Path, checkpoint_sha256: str) -> tuple[str, dict[str, Any]]:
    parameters = []
    for name, value in model.named_parameters():
        try:
            version = int(value._version)
        except Exception:
            version = -1
        parameters.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": list(value.stride()),
                "storage_offset": int(value.storage_offset()),
                "version": version,
                "data_ptr": int(value.data_ptr()),
            }
        )
    payload = {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_object_id": id(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "torch": _REQUIRED_TORCH,
        "mlx": _REQUIRED_MLX,
        "weight_mapping_version": 2,
        "weight_residency": _weight_residency(),
        "forge_adapter_sha256": _sha256(_FORGE_ROOT / "backend/modules/k_model.py"),
        "anima_mlx_sha256": _sha256(_ANIMA_MLX_SOURCE),
        "effective_model_generation": str(getattr(owner, "current_weight_patches_uuid", None)),
        "model_lowvram": bool(getattr(owner, "model_lowvram", False)),
        "lowvram_patch_counter": int(getattr(owner, "lowvram_patch_counter", 0)),
        "model_offload_buffer_memory": int(getattr(owner, "model_offload_buffer_memory", 0)),
        "parameters": parameters,
    }
    return _json_digest(payload), payload


def _walk_arrays(value: Any, array_type: type) -> Iterable[Any]:
    if isinstance(value, array_type):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk_arrays(getattr(value, field.name), array_type)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_arrays(item, array_type)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_arrays(item, array_type)


def _mapping_and_equality(model: Any, store: Any, mx: Any) -> dict[str, Any]:
    entries = []
    checks: list[tuple[str, Any]] = []
    for name, live in model.named_parameters():
        if live.device.type != "mps":
            raise _Declined(f"unsupported_offload_state:{name}:{live.device}")
        if str(live.dtype) != "torch.bfloat16":
            raise _Declined(f"unsupported_weight_dtype:{name}:{live.dtype}")
        if not live.is_contiguous() or live.storage_offset() != 0:
            raise _Declined(f"unsupported_weight_layout:{name}")
        try:
            checkpoint_weight = store.load(name)
            checkpoint_key = store.full_key(name)
        except KeyError as exc:
            raise _Declined(f"weight_mapping_missing:{name}") from exc
        if tuple(live.shape) != tuple(checkpoint_weight.shape):
            raise _Declined(f"weight_shape_mismatch:{name}")
        live_view = mx.from_dlpack(live, copy=False)
        check = mx.all(live_view == checkpoint_weight)
        checks.append((name, check))
        entries.append(
            {
                "forge_name": name,
                "checkpoint_key": checkpoint_key,
                "forge_shape": list(live.shape),
                "mlx_shape": list(checkpoint_weight.shape),
                "forge_dtype": str(live.dtype),
                "mlx_dtype": str(checkpoint_weight.dtype),
                "layout": "PyTorch nn.Linear [out,in]; MLX linear transposes weight at matmul",
                "bytes": int(checkpoint_weight.nbytes),
            }
        )

    mismatches = []
    for offset in range(0, len(checks), 24):
        chunk = checks[offset : offset + 24]
        mx.eval(*(check for _name, check in chunk))
        _synchronize(mx)
        for name, check in chunk:
            if not bool(check.item()):
                mismatches.append(name)
    if mismatches:
        preview = ",".join(mismatches[:8])
        raise _Declined(f"effective_weight_mismatch:{preview}")

    return {
        "schema_version": 1,
        "mapping_version": 1,
        "anima_mlx_source": str(_ANIMA_MLX_SOURCE),
        "anima_mlx_sha256": _sha256(_ANIMA_MLX_SOURCE),
        "parameter_count": len(entries),
        "all_live_weights_exactly_match_checkpoint": True,
        "entries": entries,
    }


class _LiveTorchWeightStore:
    """Persistent zero-copy MLX views whose original Torch owners stay live."""

    def __init__(self, model: Any, torch: Any, mx: Any):
        self.torch = torch
        self.mx = mx
        self.parameters = dict(model.named_parameters())
        self.owners = list(self.parameters.values())
        self.views: dict[str, Any] = {}

    def full_key(self, key: str) -> str:
        if key not in self.parameters:
            raise KeyError(key)
        return key

    def load(self, key: str):
        key = self.full_key(key)
        value = self.parameters[key]
        if value.device.type != "mps":
            raise _Declined(f"unsupported_offload_state:{key}:{value.device}")
        if value.dtype is not self.torch.bfloat16:
            raise _Declined(f"unsupported_weight_dtype:{key}:{value.dtype}")
        if not value.is_contiguous() or value.storage_offset() != 0:
            raise _Declined(f"unsupported_weight_layout:{key}")
        view = self.views.get(key)
        if view is None:
            view = self.mx.from_dlpack(value, copy=False)
            self.views[key] = view
        return view


def _live_weight_mapping(model: Any, store: _LiveTorchWeightStore) -> dict[str, Any]:
    entries = []
    for name, value in model.named_parameters():
        view = store.load(name)
        entries.append(
            {
                "forge_name": name,
                "checkpoint_key": name,
                "forge_shape": list(value.shape),
                "mlx_shape": list(view.shape),
                "forge_dtype": str(value.dtype),
                "mlx_dtype": str(view.dtype),
                "layout": "persistent zero-copy live Forge [out,in] view; MLX linear transposes logically",
                "bytes": int(value.numel() * value.element_size()),
                "torch_data_ptr": int(value.data_ptr()),
                "torch_version": int(getattr(value, "_version", -1)),
                "torch_owner_id": id(value),
                "mlx_view_id": id(view),
            }
        )
    return {
        "schema_version": 1,
        "mapping_version": 2,
        "anima_mlx_source": str(_ANIMA_MLX_SOURCE),
        "anima_mlx_sha256": _sha256(_ANIMA_MLX_SOURCE),
        "parameter_count": len(entries),
        "all_live_weights_exactly_match_checkpoint": None,
        "uses_live_effective_forge_weights": True,
        "copied_weight_bytes": 0,
        "entries": entries,
    }


def _write_mapping_report(report: dict[str, Any]) -> None:
    destination = Path(os.getenv("FORGE_APPLE_MLX_MAPPING_REPORT", str(_DEFAULT_MAPPING_REPORT))).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _load_state(owner: Any, model: Any, fingerprint: str, payload: dict[str, Any], checkpoint: Path, checkpoint_sha256: str) -> _ResidentState:
    torch, mx, anima_mlx = _load_runtime()
    started = time.perf_counter()
    anima_mlx.set_gelu_backend("manual")
    anima_mlx.set_attention_projection_backend("split")
    anima_mlx.set_rope_backend("metal")
    anima_mlx.set_mlp_backend("mlx")

    residency = _weight_residency()
    if residency == "shared-forge":
        store = _LiveTorchWeightStore(model, torch, mx)
        mapping = _live_weight_mapping(model, store)
        weight_owners = store.owners
        copied_weight_bytes = 0
    elif residency == "checkpoint-copy":
        store = anima_mlx.SafeTensorWeightStore(checkpoint)
        mapping = _mapping_and_equality(model, store, mx)
        weight_owners = []
        copied_weight_bytes = None
    else:
        raise _Declined(f"unsupported_weight_residency:{residency}")
    mapping.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "model_fingerprint": fingerprint,
            "torch": str(torch.__version__),
            "mlx": metadata.version("mlx"),
        }
    )
    weights = anima_mlx.load_anima(store, num_blocks=28)
    arrays = []
    seen = set()
    for value in _walk_arrays(weights, mx.array):
        marker = id(value)
        if marker not in seen:
            seen.add(marker)
            arrays.append(value)
    for offset in range(0, len(arrays), 32):
        mx.eval(*arrays[offset : offset + 32])
    _synchronize(mx)
    weight_bytes = sum(int(value.nbytes) for value in arrays)
    if copied_weight_bytes is None:
        copied_weight_bytes = weight_bytes
    shared_weight_bytes = weight_bytes if residency == "shared-forge" else 0

    def forward(latent, timestep, conditioning):
        return anima_mlx.anima_forward(latent, timestep, conditioning, weights)

    graph_started = time.perf_counter()
    compiled = mx.compile(forward, shapeless=False)
    graph_seconds = time.perf_counter() - graph_started
    load_seconds = time.perf_counter() - started
    _write_mapping_report(mapping)
    return _ResidentState(
        fingerprint=fingerprint,
        fingerprint_payload=payload,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        weights=weights,
        compiled=compiled,
        weight_arrays=arrays,
        weight_owners=weight_owners,
        weight_bytes=weight_bytes,
        copied_weight_bytes=copied_weight_bytes,
        shared_weight_bytes=shared_weight_bytes,
        weight_residency=residency,
        load_seconds=load_seconds,
        graph_construction_seconds=graph_seconds,
        mapping_report=mapping,
    )


def _release_state() -> None:
    global _STATE
    if _STATE is None:
        return
    state = _STATE
    try:
        _torch, mx, _anima_mlx = _load_runtime()
        # Complete every command that may reference foreign Torch-owned views
        # before dropping the compiled closure, MLX views, or original owners.
        _synchronize(mx)
    except Exception:
        mx = None
    _STATE = None
    del state
    gc.collect()
    if mx is not None:
        try:
            mx.clear_cache()
            _synchronize(mx)
        except Exception:
            pass


def release() -> None:
    with _LOCK:
        _release_state()
        _STATS["resident_weight_bytes"] = 0
        _STATS["copied_weight_bytes"] = 0
        _STATS["shared_weight_bytes"] = 0


def _state_for(owner: Any, model: Any) -> tuple[_ResidentState, Any, Any]:
    global _STATE
    torch, mx, _anima_mlx = _load_runtime()
    configured = os.getenv("FORGE_APPLE_MLX_DENOISER_CHECKPOINT", "").strip()
    if not configured:
        raise _Declined("missing_checkpoint_setting")
    checkpoint = Path(configured).expanduser().resolve()
    if not checkpoint.is_file():
        raise _Declined("checkpoint_not_found")
    checkpoint_sha256 = _sha256(checkpoint)
    fingerprint, payload = _model_fingerprint(owner, model, checkpoint, checkpoint_sha256)
    rejected = _REJECTED.get(fingerprint)
    if rejected is not None:
        raise _Declined(rejected)
    if _STATE is not None and _STATE.fingerprint == fingerprint:
        return _STATE, torch, mx
    if _STATE is not None:
        _STATS["state_invalidations"] += 1
        _release_state()
    try:
        _STATE = _load_state(owner, model, fingerprint, payload, checkpoint, checkpoint_sha256)
    except _Declined as exc:
        _REJECTED[fingerprint] = exc.reason
        raise
    _STATS["state_loads"] += 1
    _STATS["resident_weight_bytes"] = _STATE.weight_bytes
    _STATS["copied_weight_bytes"] = _STATE.copied_weight_bytes
    _STATS["shared_weight_bytes"] = _STATE.shared_weight_bytes
    return _STATE, torch, mx


def _set_fallback(reason: str, error: str = "") -> None:
    global _LAST_ROUTE
    _STATS["fallbacks"][reason] += 1
    if error:
        _STATS["last_error"] = error
    _LAST_ROUTE = {
        "status": "fallback",
        "reason": reason,
        "backend": _BACKEND,
        "error": error,
        "synchronization": _synchronization_description(),
    }


def _preflight(model: Any, x: Any, t: Any, context: Any, *, owner: Any, control: Any, transformer_options: Any, extra_conds: dict[str, Any], torch: Any) -> None:
    if not _is_anima(model):
        raise _Declined("unsupported_model")
    if control is not None:
        raise _Declined("control_present")
    if extra_conds:
        raise _Declined(f"unsupported_extra_conditions:{','.join(sorted(extra_conds))}")
    patch_reason = _transformer_patch_reason(transformer_options)
    if patch_reason:
        raise _Declined(patch_reason)
    if bool(getattr(owner, "model_lowvram", False)) or int(getattr(owner, "lowvram_patch_counter", 0)):
        raise _Declined("unsupported_offload_state")
    checks = (
        _tensor_reason("x", x, (2, 16, 1, 128, 128), torch.bfloat16),
        _tensor_reason("t", t, (2,), torch.float32),
        _tensor_reason("context", context, (2, 1, 512, 1024), torch.bfloat16),
    )
    for reason in checks:
        if reason:
            raise _Declined(reason)


def denoiser_forward(model: Any, x: Any, t: Any, context: Any, *, owner: Any, control: Any, transformer_options: Any, extra_conds: dict[str, Any]):
    global _LAST_ROUTE
    with _LOCK:
        _STATS["calls"] += 1
        total_started = time.perf_counter()
        try:
            torch, mx, _anima_mlx = _load_runtime()
            _preflight(model, x, t, context, owner=owner, control=control, transformer_options=transformer_options, extra_conds=extra_conds, torch=torch)
            state, torch, mx = _state_for(owner, model)

            torch.mps.synchronize()
            mlx_x = mx.from_dlpack(x, copy=False)
            mlx_t = mx.from_dlpack(t, copy=False)
            mlx_context = mx.from_dlpack(context, copy=False)
            _STATS["zero_copy_input_activations"] += 3
            ownership = None
            if _lifetime_diagnostics_enabled():
                ownership = {
                    "torch_inputs": {
                        "x": {"python_id": id(x), "data_ptr": int(x.data_ptr()), "lifetime": "caller-owned"},
                        "t": {"python_id": id(t), "data_ptr": int(t.data_ptr()), "lifetime": "caller-owned"},
                        "context": {"python_id": id(context), "data_ptr": int(context.data_ptr()), "lifetime": "caller-owned"},
                    },
                    "mlx_imports": {
                        "x": {"python_id": id(mlx_x), "ownership": "zero-copy DLPack view", "lifetime": "invocation-local"},
                        "t": {"python_id": id(mlx_t), "ownership": "zero-copy DLPack view", "lifetime": "invocation-local"},
                        "context": {"python_id": id(mlx_context), "ownership": "zero-copy DLPack view", "lifetime": "invocation-local"},
                    },
                }

            signature = json.dumps(
                {
                    "x": [list(mlx_x.shape), str(mlx_x.dtype)],
                    "t": [list(mlx_t.shape), str(mlx_t.dtype)],
                    "context": [list(mlx_context.shape), str(mlx_context.dtype)],
                    "shapeless": False,
                },
                sort_keys=True,
            )
            compile_miss = signature not in state.compiled_signatures
            execution_started = time.perf_counter()
            mlx_output = state.compiled(mlx_x, mlx_t, mlx_context)
            mx.eval(mlx_output)
            _synchronize(mx)
            execution_seconds = time.perf_counter() - execution_started
            state.compiled_signatures.add(signature)
            if compile_miss:
                _STATS["compile_cache_misses"] += 1
                _STATS["cold_compile_seconds"] = execution_seconds
            else:
                _STATS["compile_cache_hits"] += 1
                _STATS["last_warm_execution_seconds"] = execution_seconds

            output = torch.from_dlpack(mlx_output)
            _STATS["zero_copy_output_activations"] += 1
            torch.mps.synchronize()
            if output.device.type != "mps" or output.dtype is not x.dtype or tuple(output.shape) != tuple(x.shape):
                raise RuntimeError(f"invalid_output_metadata:{output.device}:{output.dtype}:{tuple(output.shape)}")
            # This forces a device reduction and host scalar read.  Keep it out
            # of the successful production route; parity/lifecycle harnesses
            # may opt in explicitly when diagnosing numerical failures.
            if _diagnostic_finite_check_enabled():
                _STATS["diagnostic_finite_checks"] += 1
                if not bool(torch.isfinite(output).all().item()):
                    raise RuntimeError("nonfinite_output")
            if ownership is not None:
                ownership["mlx_output"] = {
                    "python_id": id(mlx_output),
                    "ownership": "MLX-produced DLPack exporter",
                    "lifetime": "retained by exported Torch tensor until consumer release",
                }
                ownership["torch_output"] = {
                    "python_id": id(output),
                    "data_ptr": int(output.data_ptr()),
                    "ownership": "zero-copy DLPack consumer",
                    "lifetime": "returned to Forge caller",
                }

            state.invocation_count += 1
            total_seconds = time.perf_counter() - total_started
            _STATS["hits"] += 1
            _STATS["last_total_seconds"] = total_seconds
            _LAST_ROUTE = {
                "status": "hit",
                "reason": "",
                "backend": _BACKEND,
                "model_fingerprint": state.fingerprint,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "compile_cache": "miss" if compile_miss else "hit",
                "execution_seconds": execution_seconds,
                "total_seconds": total_seconds,
                "zero_copy_input_activations": 3,
                "zero_copy_output_activations": 1,
                "copied_activation_count": 0,
                "resident_weight_bytes": state.weight_bytes,
                "copied_weight_bytes": state.copied_weight_bytes,
                "shared_weight_bytes": state.shared_weight_bytes,
                "weight_residency": state.weight_residency,
                "diagnostic_finite_check": _diagnostic_finite_check_enabled(),
                "synchronization": _synchronization_description(),
            }
            if ownership is not None:
                _LAST_ROUTE["ownership"] = ownership
            return output
        except _Declined as exc:
            _set_fallback(exc.reason)
            return None
        except Exception as exc:
            try:
                if "mx" in locals():
                    _synchronize(mx)
            except Exception:
                pass
            detail = f"{type(exc).__name__}:{exc}"
            _set_fallback("mlx_invocation_failed", detail)
            return None


def last_route() -> dict[str, Any]:
    with _LOCK:
        return dict(_LAST_ROUTE)


def _memory_status() -> dict[str, Any]:
    try:
        torch, mx, _anima_mlx = _load_runtime()
        return {
            "torch_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "torch_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}:{exc}"}


def status() -> dict[str, Any]:
    with _LOCK:
        state = _STATE
        payload = {
            "backend": _BACKEND,
            "enabled": os.getenv("FORGE_APPLE_DENOISER_BACKEND", "off").strip().lower() == _BACKEND,
            "required_torch": _REQUIRED_TORCH,
            "required_mlx": _REQUIRED_MLX,
            "mlx_site": _mlx_site(),
            "weight_residency": _weight_residency(),
            "sync_bridge": mlx_sync_bridge.status(),
            "last_route": dict(_LAST_ROUTE),
            "stats": {
                **{key: value for key, value in _STATS.items() if key != "fallbacks"},
                "fallbacks": dict(_STATS["fallbacks"]),
            },
            "memory": _memory_status(),
            "state": None,
        }
        if state is not None:
            payload["state"] = {
                "fingerprint": state.fingerprint,
                "checkpoint": str(state.checkpoint),
                "checkpoint_sha256": state.checkpoint_sha256,
                "weight_bytes": state.weight_bytes,
                "copied_weight_bytes": state.copied_weight_bytes,
                "shared_weight_bytes": state.shared_weight_bytes,
                "weight_residency": state.weight_residency,
                "retained_torch_weight_owner_count": len(state.weight_owners),
                "load_seconds": state.load_seconds,
                "graph_construction_seconds": state.graph_construction_seconds,
                "invocation_count": state.invocation_count,
                "compile_signature_count": len(state.compiled_signatures),
                "mapping_parameter_count": state.mapping_report.get("parameter_count"),
                "effective_weights_exact": state.mapping_report.get("all_live_weights_exactly_match_checkpoint"),
                "uses_live_effective_forge_weights": state.mapping_report.get("uses_live_effective_forge_weights", False),
            }
        return payload
