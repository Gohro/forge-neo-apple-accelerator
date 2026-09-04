"""Settings and runtime status for the Forge Neo Apple Accelerator."""

from __future__ import annotations

import html
import json
import os
import platform
import sys
from pathlib import Path

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, validate_manifest
from forge_apple_accelerator.provider import provider
from forge_apple_accelerator import stock_adapter
from modules import script_callbacks, shared

try:
    from modules import acceleration_providers
except ImportError:
    acceleration_providers = None


if acceleration_providers is None:
    stock_adapter.install_rng_compat(provider)
stock_adapter.apply_upscaler_settings(shared)


def registered_providers() -> list[dict]:
    if acceleration_providers is None:
        adapter = stock_adapter.status()
        return [{"name": "forge_apple_accelerator_stock_adapter", "priority": 100}] if adapter.get("installed") else []
    return acceleration_providers.registered_providers()


def last_route(operation: str) -> dict:
    if acceleration_providers is None:
        try:
            return provider.route(operation)
        except Exception:
            return {"backend": "stock_fallback", "operation": operation}
    return acceleration_providers.last_route(operation)



def selected_manifest_path() -> Path:
    configured = os.getenv("FORGE_APPLE_MPSGRAPH_ATTENTION_MANIFEST", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MANIFEST_PATH


def manifest_status() -> str:
    try:
        import torch

        valid, reason, manifest, _module_file = validate_manifest(selected_manifest_path(), torch_module=torch)
        if not valid:
            return f"native bridge unavailable ({reason})"
        return f"native bridge ready for Torch {manifest.get('torch', 'unknown')} (verified hashes)"
    except Exception as exc:
        return f"native bridge unavailable ({type(exc).__name__})"


def stable_runtime_status() -> dict:
    """Report install and activation state without importing a second Torch."""
    manifest_path = EXTENSION_ROOT / "runtime_overlay_manifest.json"
    stamp_path = EXTENSION_ROOT / "runtime_overlay" / "forge_apple_runtime.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_runtime = manifest.get("runtime_id", "unknown")
    except Exception as exc:
        return {
            "installed": False,
            "active": False,
            "reason": f"manifest_unavailable:{type(exc).__name__}",
        }
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        installed = stamp.get("runtime_id") == expected_runtime
        reason = "ready" if installed else "runtime_id_mismatch"
    except Exception as exc:
        installed = False
        reason = f"stamp_unavailable:{type(exc).__name__}"

    overlay_root = (EXTENSION_ROOT / "runtime_overlay").resolve()
    try:
        import torch

        torch_version = getattr(torch, "__long_version__", torch.__version__)
        torch_file = Path(torch.__file__).resolve()
        overlay_active = torch_file == overlay_root or overlay_root in torch_file.parents
    except Exception:
        torch_version = "unavailable"
        torch_file = None
        overlay_active = False

    return {
        "installed": installed,
        "active": bool(installed and overlay_active),
        "overlay_torch_active": overlay_active,
        "runtime_id": expected_runtime,
        "torch_version": str(torch_version),
        "torch_file": str(torch_file) if torch_file is not None else "",
        "reason": reason,
    }


def runtime_status_payload() -> dict:
    keys = (
        "FORGE_APPLE_ACCELERATOR_MODE",
        "FORGE_APPLE_BACKEND",
        "FORGE_APPLE_METAL_KERNELS",
        "FORGE_APPLE_ATTENTION_BACKEND",
        "FORGE_APPLE_ATTENTION_CROSS",
        "FORGE_APPLE_LINEAR_BACKEND",
        "FORGE_APPLE_DENOISER_BACKEND",
        "FORGE_APPLE_ANIMA_COMPAT_BACKEND",
        "FORGE_APPLE_MLX_SITE",
        "FORGE_APPLE_MLX_DENOISER_CHECKPOINT",
        "FORGE_APPLE_UPSCALER_GPU_COMPOSITE",
        "FORGE_APPLE_SWINIR_BF16",
        "FORGE_APPLE_SWINIR_COMPILE",
        "FORGE_APPLE_UPSCALER_TILE",
    )
    environment = {key: os.getenv(key, "") for key in keys}
    disabled_values = {"", "0", "false", "no", "off", "none"}
    active = {
        key: value
        for key, value in environment.items()
        if value.strip().lower() not in disabled_values
    }
    try:
        import torch

        manifest_valid, manifest_reason, manifest, module_path = validate_manifest(selected_manifest_path(), torch_module=torch)
        bridge = {
            "valid": manifest_valid,
            "reason": manifest_reason,
            "schema_version": manifest.get("schema_version"),
            "torch": manifest.get("torch"),
            "module_file": str(module_path) if module_path is not None else "",
        }
    except Exception as exc:
        bridge = {"valid": False, "reason": f"{type(exc).__name__}:{exc}"}
    return {
        "mode": environment["FORGE_APPLE_ACCELERATOR_MODE"] or "inherit",
        "classic_oracle": not active,
        "active_apple_overrides": active,
        "environment": environment,
        "registered_providers": registered_providers(),
        "stock_adapter": stock_adapter.status(),
        "routes": {
            operation: last_route(operation)
            for operation in ("rope_pair", "addcmul", "attention_core", "attention_shader", "linear", "denoiser_forward")
        },
        "provider": provider.status(),
        "bridge": bridge,
        "stable_runtime": stable_runtime_status(),
    }


def on_app_started(_demo, app) -> None:
    stock_adapter.activate_swinir_bf16()
    app.add_api_route(
        "/internal/forge-apple-accelerator/status",
        runtime_status_payload,
        methods=["GET"],
    )
    # Compatibility for the pre-extension Apple/MLX comparison harness.
    app.add_api_route(
        "/internal/apple-oracle-status",
        runtime_status_payload,
        methods=["GET"],
    )


def runtime_status_html() -> str:
    runtime = stable_runtime_status()
    if runtime["active"]:
        runtime_label = f"active ({runtime['torch_version']})"
    elif runtime["installed"]:
        runtime_label = "installed, inactive; restart Forge"
    else:
        runtime_label = "not installed; base-runtime acceleration active"
    fields = {
        "Host": f"{platform.system()} {platform.machine()}",
        "Mode": os.getenv("FORGE_APPLE_ACCELERATOR_MODE", "inherit"),
        "Metal kernels": os.getenv("FORGE_APPLE_METAL_KERNELS", "off"),
        "Attention": os.getenv("FORGE_APPLE_ATTENTION_BACKEND", "off"),
        "Cross-attention": os.getenv("FORGE_APPLE_ATTENTION_CROSS", "off"),
        "Whole denoiser": os.getenv("FORGE_APPLE_DENOISER_BACKEND", "off"),
        "Upscaler GPU composite": os.getenv("FORGE_APPLE_UPSCALER_GPU_COMPOSITE", "off"),
        "SwinIR BF16": os.getenv("FORGE_APPLE_SWINIR_BF16", "off"),
        "Compiled SwinIR": os.getenv("FORGE_APPLE_SWINIR_COMPILE", "off"),
        "Upscaler tile": os.getenv("FORGE_APPLE_UPSCALER_TILE", "256"),
        "Integration": "stock guarded adapter" if acceleration_providers is None else "provider API v1",
        "Registered providers": ", ".join(item["name"] for item in registered_providers()) or "none",
        "Bridge": manifest_status(),
        "Stable Apple runtime": runtime_label,
        "Loaded Torch": runtime["torch_version"],
    }
    rows = "".join(
        f"<tr><th style='text-align:left;padding-right:1rem'>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in fields.items()
    )
    return (
        f"<table>{rows}</table>"
        "<p>Qualified acceleration is automatic while the add-on is enabled. "
        "Disable or re-enable the add-on in Forge's Extensions panel, then restart Forge.</p>"
    )


def on_ui_settings() -> None:
    section = ("forge_apple_accelerator", "Apple Accelerator")
    status = shared.OptionHTML(runtime_status_html())
    status.section = section
    status.category_id = "system"
    shared.opts.add_option(
        "forge_apple_accelerator_status",
        status,
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
