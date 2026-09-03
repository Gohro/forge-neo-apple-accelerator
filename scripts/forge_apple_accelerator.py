"""Settings and runtime status for the Forge Neo Apple Accelerator."""

from __future__ import annotations

import html
import os
import platform
import sys
from pathlib import Path

import gradio as gr

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



def manifest_status() -> str:
    try:
        import torch

        valid, reason, manifest, _module_file = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
        if not valid:
            return f"native bridge unavailable ({reason})"
        return f"native bridge ready for Torch {manifest.get('torch', 'unknown')} (verified hashes)"
    except Exception as exc:
        return f"native bridge unavailable ({type(exc).__name__})"


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

        manifest_valid, manifest_reason, manifest, module_path = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
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
    }


def on_app_started(_demo, app) -> None:
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
    fields = {
        "Host": f"{platform.system()} {platform.machine()}",
        "Mode": os.getenv("FORGE_APPLE_ACCELERATOR_MODE", "inherit"),
        "Metal kernels": os.getenv("FORGE_APPLE_METAL_KERNELS", "off"),
        "Attention": os.getenv("FORGE_APPLE_ATTENTION_BACKEND", "off"),
        "Cross-attention": os.getenv("FORGE_APPLE_ATTENTION_CROSS", "off"),
        "Whole denoiser": os.getenv("FORGE_APPLE_DENOISER_BACKEND", "off"),
        "Upscaler GPU composite": os.getenv("FORGE_APPLE_UPSCALER_GPU_COMPOSITE", "off"),
        "Upscaler tile": os.getenv("FORGE_APPLE_UPSCALER_TILE", "256"),
        "Integration": "stock guarded adapter" if acceleration_providers is None else "provider API v1",
        "Registered providers": ", ".join(item["name"] for item in registered_providers()) or "none",
        "Bridge": manifest_status(),
    }
    rows = "".join(
        f"<tr><th style='text-align:left;padding-right:1rem'>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in fields.items()
    )
    return f"<table>{rows}</table><p>Mode changes require a full Forge restart.</p>"


def on_ui_settings() -> None:
    section = ("forge_apple_accelerator", "Apple Accelerator")
    shared.opts.add_option(
        "forge_apple_accelerator_mode",
        shared.OptionInfo(
            "visual-fast",
            "Acceleration mode",
            gr.Dropdown,
            {"choices": ["inherit", "off", "strict", "visual-fast"]},
            section=section,
            category_id="system",
        )
        .info("Strict keeps only exact promoted routes; visual-fast enables parity-gated MPSGraph attention.")
        .needs_restart(),
    )
    shared.opts.add_option(
        "forge_apple_accelerator_upscaler_gpu_composite",
        shared.OptionInfo(
            True,
            "Use GPU tile compositing for SwinIR/ESRGAN",
            section=section,
            category_id="system",
        )
        .info("Measured 29% faster for the controlled SwinIR compositor comparison.")
        .needs_restart(),
    )
    shared.opts.add_option(
        "forge_apple_accelerator_upscaler_tile",
        shared.OptionInfo(
            768,
            "Apple upscaler tile size",
            gr.Slider,
            {"minimum": 256, "maximum": 1024, "step": 16},
            section=section,
            category_id="system",
        )
        .info("768/full-model behavior is qualified on the development Mac; use 512 on lower-memory Macs.")
        .needs_restart(),
    )
    status = shared.OptionHTML(runtime_status_html())
    status.section = section
    status.category_id = "system"
    shared.opts.add_option(
        "forge_apple_accelerator_status",
        status,
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
