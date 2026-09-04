"""Guarded BF16 compatibility patch for Spandrel's SwinIR attention.

Spandrel advertises BF16 support for SwinIR, but shifted-window masks are
created in FP32.  Adding that mask promotes the attention probabilities to
FP32 and the subsequent ``attn @ v`` fails because ``v`` remains BF16.  Keep
the numerically sensitive softmax in FP32 and cast its probabilities back to
the value dtype immediately before the value matmul.

The patch is deliberately inert for every dtype other than BF16.
"""

from __future__ import annotations

import inspect
import os
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()
_ORIGINAL_FORWARD = None
_ORIGINAL_LOAD_SPANDREL_MODEL = None
_STATUS: dict[str, Any] = {
    "installed": False,
    "compatible": False,
    "reason": "not_attempted",
    "class": "",
    "loader_patched": False,
    "loader_reason": "not_attempted",
    "compile_requested": False,
    "compile_supported": False,
    "compile_active": False,
    "compile_reason": "not_requested",
    "compiled_models": 0,
    "compile_fallbacks": 0,
    "last_model": "",
    "last_model_qualified": False,
}


_FALSE_VALUES = {"", "0", "false", "no", "off", "none"}


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() not in _FALSE_VALUES


def install() -> dict[str, Any]:
    """Install the guarded class-level patch and return its status."""

    global _ORIGINAL_FORWARD
    with _LOCK:
        if _STATUS["installed"]:
            return status()

        try:
            import torch
            from spandrel.architectures.SwinIR.__arch.SwinIR import WindowAttention
        except Exception as exc:
            _STATUS.update(reason=f"import_error:{type(exc).__name__}:{exc}")
            return status()

        signature = tuple(inspect.signature(WindowAttention.forward).parameters)
        if signature != ("self", "x", "mask"):
            _STATUS.update(reason=f"unsupported_forward_signature:{signature}")
            return status()

        _ORIGINAL_FORWARD = WindowAttention.forward

        def forward(self, x, mask=None):
            if x.dtype is not torch.bfloat16:
                return _ORIGINAL_FORWARD(self, x, mask=mask)

            batch_windows, tokens, channels = x.shape
            qkv = (
                self.qkv(x)
                .reshape(
                    batch_windows,
                    tokens,
                    3,
                    self.num_heads,
                    channels // self.num_heads,
                )
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            attention = (q * self.scale) @ k.transpose(-2, -1)
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attention = attention + relative_position_bias.unsqueeze(0)

            # Preserve an FP32 softmax even though weights/activations use BF16.
            # This is both the compatibility fix and the conservative quality
            # policy; only the probability/value matmul returns to BF16.
            attention = attention.float()
            if mask is not None:
                window_count = mask.shape[0]
                attention = attention.view(
                    batch_windows // window_count,
                    window_count,
                    self.num_heads,
                    tokens,
                    tokens,
                )
                attention = attention + mask.float().unsqueeze(1).unsqueeze(0)
                attention = attention.view(-1, self.num_heads, tokens, tokens)
            attention = self.softmax(attention)
            attention = self.attn_drop(attention).to(dtype=v.dtype)

            output = (attention @ v).transpose(1, 2).reshape(
                batch_windows,
                tokens,
                channels,
            )
            output = self.proj(output)
            return self.proj_drop(output)

        WindowAttention.forward = forward
        _STATUS.update(
            {
                "installed": True,
                "compatible": True,
                "reason": "ok",
                "class": f"{WindowAttention.__module__}.{WindowAttention.__name__}",
            }
        )
        return status()


def status() -> dict[str, Any]:
    with _LOCK:
        return dict(_STATUS)


def install_loader_patch() -> dict[str, Any]:
    """Make Forge load only SwinIR descriptors in BF16.

    This intentionally does not enable Forge's global half-upscaler option,
    which could change the dtype of unrelated ESRGAN architectures.
    """

    global _ORIGINAL_LOAD_SPANDREL_MODEL
    with _LOCK:
        attention_status = install()
        if not attention_status.get("compatible"):
            return status()
        if _STATUS["loader_patched"]:
            return status()

        try:
            import torch
            from modules import modelloader
        except Exception as exc:
            _STATUS["loader_reason"] = f"import_error:{type(exc).__name__}:{exc}"
            return status()

        compile_requested = _enabled("FORGE_APPLE_SWINIR_COMPILE")
        try:
            from packaging.version import Version

            compile_supported = bool(
                compile_requested
                and hasattr(torch, "compile")
                and Version(torch.__version__.split("+", 1)[0]) == Version("2.14.0")
            )
        except Exception:
            compile_supported = False
        _STATUS.update(
            compile_requested=compile_requested,
            compile_supported=compile_supported,
            compile_reason=(
                "pending_model_load"
                if compile_supported
                else "requires_torch_2.14_or_newer"
                if compile_requested
                else "not_requested"
            ),
        )

        original = modelloader.load_spandrel_model
        signature = tuple(inspect.signature(original).parameters)
        if not {"path", "device", "prefer_half"}.issubset(signature):
            _STATUS["loader_reason"] = f"unsupported_loader_signature:{signature}"
            return status()

        _ORIGINAL_LOAD_SPANDREL_MODEL = original

        def load_spandrel_model(path, device, prefer_half=False, *args, **kwargs):
            descriptor = _ORIGINAL_LOAD_SPANDREL_MODEL(
                path,
                device,
                prefer_half=prefer_half,
                *args,
                **kwargs,
            )
            architecture = getattr(getattr(descriptor, "architecture", None), "name", "")
            model_name = Path(str(path)).name
            qualified_model = architecture == "SwinIR" and model_name.lower() == "swinir_4x.pth"
            with _LOCK:
                _STATUS.update(last_model=model_name, last_model_qualified=qualified_model)
            if qualified_model and descriptor.supports_bfloat16:
                descriptor.bfloat16()
                if compile_supported:
                    try:
                        eager_call = descriptor._call_fn
                        compiled_model = torch.compile(
                            descriptor.model,
                            backend="inductor",
                            mode="reduce-overhead",
                            fullgraph=False,
                        )

                        compiled_enabled = True

                        def call_with_eager_fallback(model, image):
                            nonlocal compiled_enabled
                            if compiled_enabled:
                                try:
                                    return eager_call(compiled_model, image)
                                except Exception as exc:
                                    compiled_enabled = False
                                    with _LOCK:
                                        _STATUS.update(
                                            compile_active=False,
                                            compile_reason=f"runtime_fallback:{type(exc).__name__}:{exc}",
                                            compile_fallbacks=int(_STATUS["compile_fallbacks"]) + 1,
                                        )
                            return eager_call(model, image)

                        # Keep Spandrel's original model as the descriptor owner.
                        # Only the call path is compiled, so dtype/device helpers
                        # remain intact and a graph failure can retry eager safely.
                        descriptor._call_fn = call_with_eager_fallback
                        with _LOCK:
                            _STATUS.update(
                                compile_active=True,
                                compile_reason="ok",
                                compiled_models=int(_STATUS["compiled_models"]) + 1,
                            )
                    except Exception as exc:
                        with _LOCK:
                            _STATUS.update(
                                compile_active=False,
                                compile_reason=f"compile_error:{type(exc).__name__}:{exc}",
                            )
            return descriptor

        modelloader.load_spandrel_model = load_spandrel_model
        _STATUS.update(loader_patched=True, loader_reason="ok")
        return status()
