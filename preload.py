"""Early bootstrap for the Forge Neo Apple Accelerator.

Forge calls extension preload hooks before shared initialization and model
backend imports. That is early enough to select guarded Apple providers through
environment variables without modifying a user's launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = Path(__file__).resolve().parent
_EXTENSION_PACKAGE_PARENT = _EXTENSION_ROOT
_MANIFEST = _EXTENSION_ROOT / "native_build/mpsgraph_attention_interop/manifest.json"
_STABLE_MANIFEST = _EXTENSION_ROOT / "native_build/mpsgraph_attention_interop-stable/manifest.json"
_RUNTIME_MANIFEST = _EXTENSION_ROOT / "runtime_overlay_manifest.json"
_RUNTIME_OVERLAY = _EXTENSION_ROOT / "runtime_overlay"
_RUNTIME_STAMP = _RUNTIME_OVERLAY / "forge_apple_runtime.json"
_VALID_MODES = {"inherit", "off", "strict", "visual-fast"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _activate_stable_runtime() -> bool:
    """Select the extension runtime before Forge imports Torch."""

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    if "torch" in sys.modules:
        os.environ["FORGE_APPLE_RUNTIME_ACTIVATION_ERROR"] = "torch_already_imported"
        return False
    try:
        manifest = json.loads(_RUNTIME_MANIFEST.read_text(encoding="utf-8"))
        stamp = json.loads(_RUNTIME_STAMP.read_text(encoding="utf-8"))
        if stamp.get("runtime_id") != manifest.get("runtime_id"):
            raise RuntimeError("runtime_id_mismatch")
        if stamp.get("manifest_sha256") != _sha256(_RUNTIME_MANIFEST):
            raise RuntimeError("manifest_hash_mismatch")
        if not _STABLE_MANIFEST.is_file():
            raise RuntimeError("stable_bridge_manifest_missing")
    except Exception as exc:
        os.environ["FORGE_APPLE_RUNTIME_ACTIVATION_ERROR"] = f"{type(exc).__name__}:{exc}"
        return False

    sys.path.insert(0, str(_RUNTIME_OVERLAY))
    os.environ["FORGE_APPLE_STABLE_RUNTIME_ACTIVE"] = "1"
    os.environ["FORGE_APPLE_MPSGRAPH_ATTENTION_MANIFEST"] = str(_STABLE_MANIFEST)
    os.environ["FORGE_APPLE_MPSGRAPH_EXTENSION_MANIFEST"] = str(_STABLE_MANIFEST)
    compile_cache = _EXTENSION_ROOT / "runtime_cache/torchinductor"
    compile_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(compile_cache))
    return True


def _recommended_upscaler_tile() -> int:
    try:
        total_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        return 512
    return 768 if total_bytes >= 32 * 1024**3 else 512


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
    # v0.2 makes extension enable/disable the public control. Ignore stale
    # pre-v0.2 performance-mode values left in config.json after an upgrade.
    return "visual-fast"


def _apply_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        mode = "inherit"
    os.environ["FORGE_APPLE_ACCELERATOR_MODE"] = mode
    if mode == "inherit":
        return

    # Core ML remains outside the promoted Forge path in every parity-first mode.
    os.environ["FORGE_APPLE_BACKEND"] = "off"
    if mode == "visual-fast":
        # Whole-denoiser MLX stays disabled unless the user explicitly opts in.
        os.environ.setdefault("FORGE_APPLE_DENOISER_BACKEND", "off")
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "rope"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "mpsgraph-sdpa"
        os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "1"
        os.environ["FORGE_APPLE_UPSCALER_GPU_COMPOSITE"] = "1"
        os.environ["FORGE_APPLE_SWINIR_BF16"] = "1"
        os.environ["FORGE_APPLE_SWINIR_COMPILE"] = (
            "1" if os.getenv("FORGE_APPLE_STABLE_RUNTIME_ACTIVE") == "1" else "off"
        )
        os.environ["FORGE_APPLE_UPSCALER_TILE"] = str(_recommended_upscaler_tile())
    elif mode == "strict":
        os.environ["FORGE_APPLE_DENOISER_BACKEND"] = "off"
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "rope"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_GPU_COMPOSITE"] = "off"
        os.environ["FORGE_APPLE_SWINIR_BF16"] = "off"
        os.environ["FORGE_APPLE_SWINIR_COMPILE"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_TILE"] = "256"
    else:
        os.environ["FORGE_APPLE_DENOISER_BACKEND"] = "off"
        os.environ["FORGE_APPLE_METAL_KERNELS"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_BACKEND"] = "off"
        os.environ["FORGE_APPLE_ATTENTION_CROSS"] = "off"
        os.environ["FORGE_APPLE_UPSCALER_GPU_COMPOSITE"] = "off"
        os.environ["FORGE_APPLE_SWINIR_BF16"] = "off"
        os.environ["FORGE_APPLE_SWINIR_COMPILE"] = "off"
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
    if supported_host:
        _activate_stable_runtime()
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
