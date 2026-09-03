"""Lazy provider adapter registered with Forge's neutral dispatch seam."""

from __future__ import annotations

import importlib
import os
import threading
from typing import Any


class AppleSiliconProvider:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._modules: dict[str, Any] = {}

    def _runtime(self, name: str):
        with self._lock:
            module = self._modules.get(name)
            if module is None:
                module = importlib.import_module(f"forge_apple_accelerator.runtime.{name}")
                self._modules[name] = module
            return module

    def enabled(self, operation: str) -> bool:
        metal_mode = os.getenv("FORGE_APPLE_METAL_KERNELS", "off").strip().lower()
        if operation == "rope_pair":
            return metal_mode in {"rope", "all", "1", "true", "yes", "on"}
        if operation == "addcmul":
            return metal_mode in {"addcmul", "pointwise", "probe", "all", "1", "true", "yes", "on"}
        if operation == "attention_core":
            backend = os.getenv("FORGE_APPLE_ATTENTION_BACKEND", "off").strip().lower()
            return backend.startswith("mpsgraph-") or backend == "torch-sdpa-self-mpsgraph-cross"
        if operation == "attention_shader":
            return metal_mode in {"attention", "all", "1", "true", "yes", "on"}
        if operation == "linear":
            return os.getenv("FORGE_APPLE_LINEAR_BACKEND", "off").strip().lower() == "mpsgraph-linear"
        if operation in {"anima_layer_norm", "anima_rms_norm", "anima_gelu", "legacy_mps_randn"}:
            return os.getenv("FORGE_APPLE_ANIMA_COMPAT_BACKEND", "off").strip().lower() == "torch212-exact"
        if operation == "denoiser_forward":
            return os.getenv("FORGE_APPLE_DENOISER_BACKEND", "off").strip().lower() == "mlx-anima-experimental"
        return False

    def rope_pair(self, xq, xk, freqs):
        return self._runtime("metal_kernels").rope_pair(xq, xk, freqs)

    def addcmul(self, input_tensor, tensor1, tensor2):
        return self._runtime("metal_kernels").addcmul(input_tensor, tensor1, tensor2)

    def attention_core(self, q, k, v, *, is_self_attention: bool):
        return self._runtime("mpsgraph_attention").attention_core(q, k, v, is_self_attention=is_self_attention)

    def attention_shader(self, q, k, v, *, is_self_attention: bool):
        return self._runtime("metal_kernels").attention_core(q, k, v, is_self_attention=is_self_attention)

    def linear(self, input_tensor, weight, *, kind: str = ""):
        return self._runtime("mpsgraph_linear").linear(input_tensor, weight, kind=kind)

    def anima_layer_norm(self, input_tensor, *, normalized_shape, eps: float, elementwise_affine: bool):
        return self._runtime("anima_compat").layer_norm(
            input_tensor,
            normalized_shape=tuple(normalized_shape),
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def anima_rms_norm(self, input_tensor, weight, *, normalized_shape, eps: float):
        return self._runtime("anima_compat").rms_norm(
            input_tensor,
            weight,
            normalized_shape=tuple(normalized_shape),
            eps=eps,
        )

    def anima_gelu(self, input_tensor, *, approximate: str):
        return self._runtime("anima_compat").gelu(input_tensor, approximate=approximate)

    def legacy_mps_randn(self, shape, *, generator=None):
        return self._runtime("anima_compat").legacy_randn(tuple(shape), generator=generator)

    def denoiser_forward(self, diffusion_model, x, t, context, *, owner, control, transformer_options, extra_conds):
        return self._runtime("mlx_anima_denoiser").denoiser_forward(
            diffusion_model,
            x,
            t,
            context,
            owner=owner,
            control=control,
            transformer_options=transformer_options,
            extra_conds=extra_conds,
        )

    def route(self, operation: str) -> dict[str, Any]:
        if operation == "attention_core":
            return self._runtime("mpsgraph_attention").last_route()
        if operation == "denoiser_forward":
            return self._runtime("mlx_anima_denoiser").last_route()
        if operation in {"anima_layer_norm", "anima_rms_norm", "anima_gelu", "legacy_mps_randn"}:
            return self._runtime("anima_compat").last_route()
        return {"backend": "forge_apple_accelerator"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = dict(self._modules)
        payload: dict[str, Any] = {"loaded_runtime_modules": sorted(loaded)}
        metal = loaded.get("metal_kernels")
        if metal is not None:
            payload["metal"] = {"mode": metal.mode(), "stats": metal.stats()}
        attention = loaded.get("mpsgraph_attention")
        if attention is not None:
            payload["attention"] = {
                "last_route": attention.last_route(),
                "last_error": attention.last_error(),
            }
        linear = loaded.get("mpsgraph_linear")
        if linear is not None:
            payload["linear"] = linear.stats()
        compat = loaded.get("anima_compat")
        if compat is not None:
            payload["anima_compat"] = compat.stats()
        denoiser = loaded.get("mlx_anima_denoiser")
        if denoiser is not None:
            payload["denoiser"] = denoiser.status()
        return payload


provider = AppleSiliconProvider()
