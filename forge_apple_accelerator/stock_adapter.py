"""Guarded stock-Forge adapter for the Apple accelerator.

Forge Neo does not currently expose the experimental acceleration-provider
registry used by the research checkout.  This module installs the smallest
equivalent hooks directly on the stock Anima classes.  Every hook is guarded by
the expected public class/method shape and falls back to the original Forge
implementation when a native route declines an input.
"""

from __future__ import annotations

import inspect
import os
import threading
import types
from typing import Any


_LOCK = threading.RLock()
_ORIGINALS: dict[str, Any] = {}
_STATUS: dict[str, Any] = {
    "adapter": "stock-v1",
    "installed": False,
    "compatible": False,
    "reason": "not_attempted",
    "patched": [],
    "rng_patched": False,
    "upscaler_settings_applied": False,
}


def _exact_compat_enabled() -> bool:
    return os.getenv("FORGE_APPLE_ANIMA_COMPAT_BACKEND", "off").strip().lower() == "torch212-exact"


def _hybrid_attention_enabled() -> bool:
    return os.getenv("FORGE_APPLE_ATTENTION_BACKEND", "off").strip().lower() == "torch-sdpa-self-mpsgraph-cross"


def _signature_names(function) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def _structural_guard(anima) -> tuple[bool, str]:
    required = ("GPT2FeedForward", "SelfCrossAttention", "FinalLayer", "Block", "Anima")
    missing = [name for name in required if not hasattr(anima, name)]
    if missing:
        return False, f"missing_classes:{','.join(missing)}"

    qkv = _signature_names(anima.SelfCrossAttention.compute_qkv)
    attention = _signature_names(anima.SelfCrossAttention.compute_attention)
    feed_forward = _signature_names(anima.GPT2FeedForward.forward)
    if qkv != ("self", "x", "context", "rope_emb"):
        return False, f"unsupported_compute_qkv_signature:{qkv}"
    if attention != ("self", "q", "k", "v", "transformer_options"):
        return False, f"unsupported_compute_attention_signature:{attention}"
    if feed_forward != ("self", "x"):
        return False, f"unsupported_feed_forward_signature:{feed_forward}"
    if not all(hasattr(anima.SelfCrossAttention, name) for name in ("torch_attention_op", "forward")):
        return False, "missing_attention_methods"
    return True, "ok"


def _try_rms_norm(provider, tensor, norm):
    if not _exact_compat_enabled():
        return None
    try:
        return provider.anima_rms_norm(
            tensor,
            norm.weight,
            normalized_shape=tuple(norm.normalized_shape),
            eps=float(norm.eps),
        )
    except Exception:
        return None


def _try_layer_norm(provider, tensor, norm):
    if not _exact_compat_enabled():
        return None
    try:
        return provider.anima_layer_norm(
            tensor,
            normalized_shape=tuple(norm.normalized_shape),
            eps=float(norm.eps),
            elementwise_affine=bool(norm.elementwise_affine),
        )
    except Exception:
        return None


def _try_gelu(provider, tensor, activation):
    if not _exact_compat_enabled():
        return None
    try:
        return provider.anima_gelu(tensor, approximate=str(activation.approximate))
    except Exception:
        return None


def _try_rope(provider, q, k, rope_emb):
    try:
        return provider.rope_pair(q, k, rope_emb)
    except Exception:
        return None


def _try_attention(provider, q, k, v, *, is_self_attention: bool):
    try:
        return provider.attention_core(q, k, v, is_self_attention=is_self_attention)
    except Exception:
        return None


def _try_nightly_self_attention(torch, q, k, v):
    if not _hybrid_attention_enabled():
        return None
    if q.shape != (2, 4096, 16, 128) or k.shape != q.shape or v.shape != q.shape:
        return None
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if q.device.type != "mps" or k.device.type != "mps" or v.device.type != "mps":
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return None
    q_bhsd = q.permute(0, 2, 1, 3)
    k_bhsd = k.permute(0, 2, 1, 3)
    v_bhsd = v.permute(0, 2, 1, 3)
    output = torch.nn.functional.scaled_dot_product_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).reshape(2, 4096, 16 * 128)


def install(provider) -> dict[str, Any]:
    """Install guarded Anima hooks when Forge lacks provider API v1."""

    with _LOCK:
        if _STATUS["installed"]:
            return status()

        try:
            import comfy_kitchen as ck
            import torch
            from einops import rearrange
            from backend.nn import anima
        except Exception as exc:
            _STATUS.update(compatible=False, reason=f"import_error:{type(exc).__name__}:{exc}")
            return status()

        compatible, reason = _structural_guard(anima)
        if not compatible:
            _STATUS.update(compatible=False, reason=reason)
            return status()

        original_qkv = anima.SelfCrossAttention.compute_qkv
        original_attention = anima.SelfCrossAttention.compute_attention
        _ORIGINALS["SelfCrossAttention.compute_qkv"] = original_qkv
        _ORIGINALS["SelfCrossAttention.compute_attention"] = original_attention

        def compute_qkv(self, x, context=None, rope_emb=None):
            if torch.jit.is_tracing():
                return original_qkv(self, x, context, rope_emb=rope_emb)
            source = x if context is None else context
            q = self.q_proj(x)
            k = self.k_proj(source)
            v = self.v_proj(source)
            q, k, v = map(
                lambda tensor: rearrange(
                    tensor,
                    "b ... (h d) -> b ... h d",
                    h=self.n_heads,
                    d=self.head_dim,
                ),
                (q, k, v),
            )
            routed_q = _try_rms_norm(provider, q, self.q_norm)
            routed_k = _try_rms_norm(provider, k, self.k_norm)
            q = self.q_norm(q) if routed_q is None else routed_q
            k = self.k_norm(k) if routed_k is None else routed_k
            v = self.v_norm(v)
            if self.is_SelfAttn and rope_emb is not None:
                routed_rope = _try_rope(provider, q, k, rope_emb)
                q, k = ck.apply_rope_split_half(q, k, rope_emb) if routed_rope is None else routed_rope
            return q, k, v

        def compute_attention(self, q, k, v, transformer_options={}):
            if torch.jit.is_tracing():
                return original_attention(self, q, k, v, transformer_options=transformer_options)
            result = _try_attention(provider, q, k, v, is_self_attention=self.is_SelfAttn)
            if result is None and self.is_SelfAttn:
                result = _try_nightly_self_attention(torch, q, k, v)
            if result is None:
                result = self.torch_attention_op(q, k, v, transformer_options=transformer_options)
            return self.output_dropout(self.output_proj(result))

        anima.SelfCrossAttention.compute_qkv = compute_qkv
        anima.SelfCrossAttention.compute_attention = compute_attention
        patched = [
            "SelfCrossAttention.compute_qkv",
            "SelfCrossAttention.compute_attention",
        ]

        if _exact_compat_enabled():
            original_ff = anima.GPT2FeedForward.forward
            original_block_fn = anima.Block._fn
            original_final_fn = anima.FinalLayer._fn
            original_anima_init = anima.Anima.__init__
            _ORIGINALS.update(
                {
                    "GPT2FeedForward.forward": original_ff,
                    "Block._fn": original_block_fn,
                    "FinalLayer._fn": original_final_fn,
                    "Anima.__init__": original_anima_init,
                }
            )

            def feed_forward(self, x):
                x = self.layer1(x)
                routed = _try_gelu(provider, x, self.activation)
                x = self.activation(x) if routed is None else routed
                return self.layer2(x)

            def layer_norm_fn(x, norm, scale, shift):
                routed = _try_layer_norm(provider, x, norm)
                normalized = norm(x) if routed is None else routed
                return normalized * (1 + scale) + shift

            def anima_init(self, *args, **kwargs):
                original_anima_init(self, *args, **kwargs)
                norm = self.t_embedding_norm
                original_norm_forward = norm.forward

                def norm_forward(instance, x):
                    routed = _try_rms_norm(provider, x, instance)
                    return original_norm_forward(x) if routed is None else routed

                norm.forward = types.MethodType(norm_forward, norm)

            anima.GPT2FeedForward.forward = feed_forward
            anima.Block._fn = staticmethod(layer_norm_fn)
            anima.FinalLayer._fn = staticmethod(layer_norm_fn)
            anima.Anima.__init__ = anima_init
            patched.extend(
                [
                    "GPT2FeedForward.forward",
                    "Block._fn",
                    "FinalLayer._fn",
                    "Anima.__init__",
                ]
            )

        _STATUS.update(
            installed=True,
            compatible=True,
            reason="ok",
            patched=patched,
            exact_compat=_exact_compat_enabled(),
        )
        return status()


def install_rng_compat(provider) -> dict[str, Any]:
    """Install the exact-nightly legacy MPS RNG route after shared init."""

    with _LOCK:
        if _STATUS["rng_patched"] or not _exact_compat_enabled():
            return status()
        try:
            import torch
            from modules import devices, rng, shared
        except Exception as exc:
            _STATUS["rng_reason"] = f"import_error:{type(exc).__name__}:{exc}"
            return status()

        original_randn = rng.randn
        original_randn_local = rng.randn_local
        original_randn_like = rng.randn_like
        original_randn_without_seed = rng.randn_without_seed
        original_create_generator = rng.create_generator
        _ORIGINALS["rng.randn"] = original_randn
        _ORIGINALS["rng.randn_local"] = original_randn_local
        _ORIGINALS["rng.randn_like"] = original_randn_like
        _ORIGINALS["rng.create_generator"] = original_create_generator
        _ORIGINALS["rng.randn_without_seed"] = original_randn_without_seed

        def legacy_normal(shape, generator=None):
            try:
                return provider.legacy_mps_randn(
                    tuple(int(value) for value in shape),
                    generator=generator,
                )
            except Exception:
                return None

        def randn(seed, shape, generator=None):
            if generator is not None:
                rng.manual_seed((seed + 100000) % 65536)
            else:
                rng.manual_seed(seed)
            if shared.opts.randn_source == "NV":
                return torch.asarray((generator or rng.nv_rng).randn(shape), device=devices.device)
            if shared.opts.randn_source == "CPU":
                return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
            if devices.device.type == "mps":
                routed = legacy_normal(shape, generator=generator)
                if routed is not None:
                    return routed
            return torch.randn(shape, device=devices.device, generator=generator)

        def randn_local(seed, shape):
            if shared.opts.randn_source == "NV":
                local_rng = rng.rng_philox.Generator(seed)
                return torch.asarray(local_rng.randn(shape), device=devices.device)
            local_device = devices.cpu if shared.opts.randn_source == "CPU" else devices.device
            generator = torch.Generator(local_device).manual_seed(int(seed))
            if shared.opts.randn_source == "GPU" and devices.device.type == "mps":
                routed = legacy_normal(shape, generator=generator)
                if routed is not None:
                    return routed
            return torch.randn(shape, device=local_device, generator=generator).to(devices.device)

        def randn_like(tensor):
            if shared.opts.randn_source == "NV":
                return torch.asarray(rng.nv_rng.randn(tensor.shape), device=tensor.device, dtype=tensor.dtype)
            if shared.opts.randn_source == "CPU":
                return torch.randn_like(tensor, device=devices.cpu).to(tensor.device)
            return torch.randn_like(tensor)

        def create_generator(seed):
            if shared.opts.randn_source == "GPU" and devices.device.type == "mps":
                return torch.Generator(devices.device).manual_seed(int(seed))
            return original_create_generator(seed)

        def randn_without_seed(shape, generator=None):
            if shared.opts.randn_source == "GPU" and devices.device.type == "mps":
                routed = legacy_normal(shape, generator=generator)
                if routed is not None:
                    return routed
            return original_randn_without_seed(shape, generator=generator)

        rng.randn = randn
        rng.randn_local = randn_local
        rng.randn_like = randn_like
        rng.create_generator = create_generator
        rng.randn_without_seed = randn_without_seed
        devices.randn = randn
        devices.randn_local = randn_local
        devices.randn_like = randn_like
        devices.randn_without_seed = randn_without_seed
        _STATUS.update(rng_patched=True, rng_reason="ok")
        return status()


def apply_upscaler_settings(shared) -> dict[str, Any]:
    """Apply extension-owned settings to stock Neo's built-in tile compositor."""

    with _LOCK:
        mode = os.getenv("FORGE_APPLE_ACCELERATOR_MODE", "off").strip().lower()
        if mode not in {"visual-fast", "inherit"}:
            return status()
        gpu_value = os.getenv("FORGE_APPLE_UPSCALER_GPU_COMPOSITE", "off").strip().lower()
        gpu_enabled = gpu_value not in {"", "0", "false", "no", "off"}
        bf16_value = os.getenv("FORGE_APPLE_SWINIR_BF16", "off").strip().lower()
        bf16_enabled = bf16_value not in {"", "0", "false", "no", "off"}
        try:
            tile = int(os.getenv("FORGE_APPLE_UPSCALER_TILE", "512"))
        except ValueError:
            tile = 512
        tile = max(0, min(tile, 1024))
        try:
            shared.opts.composite_tiles_on_gpu = gpu_enabled
            shared.opts.ESRGAN_tile = tile
        except Exception as exc:
            _STATUS["upscaler_reason"] = f"settings_error:{type(exc).__name__}:{exc}"
            return status()
        _STATUS.update(
            upscaler_settings_applied=True,
            upscaler_reason="ok",
            upscaler_gpu_composite=gpu_enabled,
            upscaler_tile=tile,
            swinir_bf16_requested=bf16_enabled,
            swinir_bf16_active=False,
            swinir_bf16_reason="pending_app_start" if bf16_enabled else "disabled",
        )
        return status()


def activate_swinir_bf16() -> dict[str, Any]:
    """Install the deferred SwinIR loader patch after Forge imports settle."""

    with _LOCK:
        requested = bool(_STATUS.get("swinir_bf16_requested"))
        if not requested:
            return status()
        try:
            from forge_apple_accelerator.runtime import swinir_bf16

            bf16_status = swinir_bf16.install_loader_patch()
            _STATUS.update(
                swinir_bf16_active=bool(bf16_status.get("loader_patched")),
                swinir_bf16_reason=bf16_status.get("loader_reason", "unknown"),
            )
        except Exception as exc:
            _STATUS.update(
                swinir_bf16_active=False,
                swinir_bf16_reason=f"install_error:{type(exc).__name__}:{exc}",
            )
        return status()


def status() -> dict[str, Any]:
    with _LOCK:
        result = dict(_STATUS)
    if result.get("swinir_bf16_requested"):
        try:
            from forge_apple_accelerator.runtime import swinir_bf16

            live = swinir_bf16.status()
            result.update(
                swinir_compile_requested=bool(live.get("compile_requested")),
                swinir_compile_supported=bool(live.get("compile_supported")),
                swinir_compile_active=bool(live.get("compile_active")),
                swinir_compile_reason=live.get("compile_reason", "unknown"),
                swinir_compiled_models=int(live.get("compiled_models", 0)),
            )
        except Exception as exc:
            result.update(
                swinir_compile_active=False,
                swinir_compile_reason=f"status_error:{type(exc).__name__}:{exc}",
            )
    return result
