"""Early bootstrap for the Forge Neo Apple Accelerator.

Forge calls extension preload hooks before shared initialization and model
backend imports. That is early enough to select guarded Apple providers through
environment variables without modifying a user's launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = Path(__file__).resolve().parent
_EXTENSION_PACKAGE_PARENT = _EXTENSION_ROOT
_MANIFEST = _EXTENSION_ROOT / "native_build/mpsgraph_attention_interop/manifest.json"
_VALID_MODES = {"inherit", "off", "strict", "visual-fast"}


def _saved_config() -> dict:
    config_path = _ROOT / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _saved_mode() -> str:
    configured = os.getenv("FORGE_APPLE_ACCELERATOR_MODE", "").strip().lower()
    if configured:
        return configured
    value = _saved_config().get("forge_apple_accelerator_mode", "visual-fast")
    normalized = str(value).strip().lower()
    return normalized if normalized in _VALID_MODES else "visual-fast"


def _apply_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        mode = "inherit"
    os.environ["FORGE_APPLE_ACCELERATOR_MODE"] = mode
    if mode == "inherit":
        return

    # Core ML remains outside the promoted Forge path in every parity-first mode.
    os.environ["FORGE_APPLE_BACKEND"] = "off"
    if mode == "visual-fast":
        saved = _saved_config()
        # Whole-denoiser MLX stays disabled unless the user explicitly opts in.
        os.environ.setdefault("FORGE_APPLE_DENOISER_BACKEND", "off")
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "rope"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "mpsgraph-sdpa"
        os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "1"
        gpu_composite = saved.get("forge_apple_accelerator_upscaler_gpu_composite", True)
        tile = saved.get("forge_apple_accelerator_upscaler_tile", 768)
        os.environ["FORGE_APPLE_UPSCALER_GPU_COMPOSITE"] = "1" if bool(gpu_composite) else "off"
        try:
            normalized_tile = max(0, min(int(tile), 1024))
        except (TypeError, ValueError):
            normalized_tile = 768
        os.environ["FORGE_APPLE_UPSCALER_TILE"] = str(normalized_tile)
    elif mode == "strict":
        os.environ["FORGE_APPLE_DENOISER_BACKEND"] = "off"
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "rope"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_GPU_COMPOSITE"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_TILE"] = "256"
    else:
        os.environ["FORGE_APPLE_DENOISER_BACKEND"] = "off"
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_GPU_COMPOSITE"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_TILE"] = "256"


def _register_provider() -> None:
    package_parent = str(_EXTENSION_PACKAGE_PARENT)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    os.environ.setdefault("FORGE_APPLE_MPSGRAPH_ATTENTION_MANIFEST", str(_MANIFEST))
    os.environ.setdefault("FORGE_APPLE_MPSGRAPH_EXTENSION_MANIFEST", str(_MANIFEST))

    from forge_apple_accelerator.provider import provider

    try:
        from modules import acceleration_providers
    except ImportError:
        from forge_apple_accelerator import stock_adapter

        adapter_status = stock_adapter.install(provider)
        if not adapter_status.get("installed"):
            print(
                "Forge Apple Accelerator: stock adapter unavailable; guarded fallbacks remain active "
                f"({adapter_status.get('reason', 'unknown')})."
            )
        else:
            print("Forge Apple Accelerator: installed guarded stock Forge Neo adapter.")
        return

    if acceleration_providers.API_VERSION != 1:
        raise RuntimeError(
            "Forge Apple Accelerator requires acceleration provider API version 1; "
            f"found {acceleration_providers.API_VERSION}"
        )
    acceleration_providers.register_provider(
        "forge_apple_accelerator",
        provider,
        priority=100,
        replace=True,
    )


class _ModeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _apply_mode(values)
        setattr(namespace, self.dest, values)


def preload(parser) -> None:
    supported_host = platform.system() == "Darwin" and platform.machine() == "arm64"
    mode = _saved_mode() if supported_host else "off"
    _apply_mode(mode)
    if supported_host:
        _register_provider()
    parser.add_argument(
        "--forge-apple-accelerator-mode",
        choices=sorted(_VALID_MODES),
        default=mode,
        action=_ModeAction,
        help="Apple Accelerator mode: off, strict exact routes, visual-fast, or inherit launcher settings.",
    )
