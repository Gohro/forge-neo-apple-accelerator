"""Exact-runtime manifest validation for the private PyTorch/MPS bridge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
EXTENSION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = EXTENSION_ROOT / "native/mpsgraph_attention_ext.mm"
DEFAULT_MANIFEST_PATH = EXTENSION_ROOT / "native_build/mpsgraph_attention_interop/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_compatibility(torch_module=None) -> dict[str, str]:
    if torch_module is None:
        import torch as torch_module

    # Forge normalizes development builds such as ``2.15.0.devYYYYMMDD`` to
    # ``2.15.0`` during startup so third-party version parsers keep working.
    # It preserves the original build identity in ``__long_version__``.  The
    # native bridge must remain pinned to that exact identity, otherwise an
    # extension compiled for one nightly could be loaded by another.
    torch_version = str(getattr(torch_module, "__long_version__", torch_module.__version__))
    return {
        "torch_version": torch_version,
        "python_version": platform.python_version(),
        "python_cache_tag": str(getattr(sys.implementation, "cache_tag", "")),
        "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
    }


def forge_adapter_fingerprint() -> dict[str, Any]:
    forge_root = None
    for parent in (EXTENSION_ROOT, *EXTENSION_ROOT.parents):
        if (parent / "modules/acceleration_providers.py").is_file():
            forge_root = parent
            break
    if forge_root is None:
        return {
            "api_version": 1,
            "registry_file": "",
            "registry_sha256": "",
            "integration_file": "",
            "integration_sha256": "",
            "denoiser_integration_file": "",
            "denoiser_integration_sha256": "",
        }
    registry = forge_root / "modules/acceleration_providers.py"
    integration = forge_root / "backend/nn/anima.py"
    denoiser_integration = forge_root / "backend/modules/k_model.py"
    return {
        "api_version": 1,
        "registry_file": str(registry),
        "registry_sha256": sha256_file(registry),
        "integration_file": str(integration) if integration.is_file() else "",
        "integration_sha256": sha256_file(integration) if integration.is_file() else "",
        "denoiser_integration_file": str(denoiser_integration) if denoiser_integration.is_file() else "",
        "denoiser_integration_sha256": sha256_file(denoiser_integration) if denoiser_integration.is_file() else "",
    }


def extension_runtime_fingerprint() -> dict[str, str]:
    files = (
        EXTENSION_ROOT / "early_bootstrap.py",
        EXTENSION_ROOT / "preload.py",
        EXTENSION_ROOT / "forge_apple_accelerator/provider.py",
        EXTENSION_ROOT / "forge_apple_accelerator/stock_adapter.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/metal_kernels.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/mpsgraph_attention.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/mpsgraph_linear.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/anima_compat.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/mlx_anima_denoiser.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/swinir_bf16.py",
    )
    return {
        str(path.relative_to(EXTENSION_ROOT)): sha256_file(path)
        for path in files
    }


def resolve_module_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    configured = manifest.get("module_file") or manifest.get("module", {}).get("file")
    if not configured:
        raise KeyError("module_file")
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    torch_module=None,
    source_path: Path = SOURCE_PATH,
) -> tuple[bool, str, dict[str, Any], Path | None]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest_unreadable:{type(exc).__name__}:{exc}", {}, None

    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False, "manifest_schema_mismatch", manifest, None
    if manifest.get("ok") is not True:
        return False, "manifest_not_ok", manifest, None

    expected = runtime_compatibility(torch_module)
    actual = manifest.get("compatibility", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            return False, f"compatibility_mismatch:{key}", manifest, None

    expected_adapter = forge_adapter_fingerprint()
    actual_adapter = manifest.get("forge_adapter", {})
    for key in ("api_version", "registry_sha256", "integration_sha256", "denoiser_integration_sha256"):
        if actual_adapter.get(key) != expected_adapter.get(key):
            return False, f"forge_adapter_mismatch:{key}", manifest, None

    if manifest.get("extension_runtime") != extension_runtime_fingerprint():
        return False, "extension_runtime_hash_mismatch", manifest, None

    try:
        expected_source_hash = sha256_file(source_path)
    except Exception as exc:
        return False, f"source_unreadable:{type(exc).__name__}:{exc}", manifest, None
    if manifest.get("source", {}).get("sha256") != expected_source_hash:
        return False, "source_hash_mismatch", manifest, None

    try:
        module_path = resolve_module_path(manifest_path, manifest)
    except Exception as exc:
        return False, f"module_path_invalid:{type(exc).__name__}:{exc}", manifest, None
    if not module_path.is_file():
        return False, "module_missing", manifest, module_path
    try:
        module_hash = sha256_file(module_path)
    except Exception as exc:
        return False, f"module_unreadable:{type(exc).__name__}:{exc}", manifest, module_path
    if manifest.get("module", {}).get("sha256") != module_hash:
        return False, "module_hash_mismatch", manifest, module_path
    return True, "ok", manifest, module_path


def import_bridge(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    required_attributes: Iterable[str] = (),
    torch_module=None,
):
    valid, reason, manifest, module_path = validate_manifest(manifest_path, torch_module=torch_module)
    if not valid or module_path is None:
        raise RuntimeError(reason)
    module_name = manifest.get("module_name") or manifest.get("module", {}).get("name")
    if not module_name:
        raise RuntimeError("module_name_missing")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module_spec_unavailable:{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in required_attributes if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"module_attributes_missing:{','.join(missing)}")
    return module, manifest
