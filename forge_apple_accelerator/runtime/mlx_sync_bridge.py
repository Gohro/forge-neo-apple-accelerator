"""Manifest-gated MLX synchronization for the Phase 2C lifecycle salvage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import sysconfig
import threading
from importlib import metadata
from pathlib import Path
from typing import Any


BACKEND_PYTHON = "python"
BACKEND_GIL_SAFE_NATIVE = "gil-safe-native"
REQUIRED_MLX = "0.32.2"
SCHEMA_VERSION = 1
EXTENSION_ROOT = Path(__file__).resolve().parents[2]
FORGE_ROOT = EXTENSION_ROOT.parents[1]
SOURCE_PATH = EXTENSION_ROOT / "native/mlx_gil_sync_ext.cpp"
DEFAULT_MANIFEST_PATH = EXTENSION_ROOT / "native_build/mlx_gil_sync/manifest.json"
_REQUIRED_SYMBOL = "__ZN3mlx4core11synchronizeEv"

_LOAD_LOCK = threading.Lock()
_BRIDGE: Any | None = None
_MANIFEST: dict[str, Any] | None = None
_LOAD_ERROR = ""


class BridgeUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_compatibility() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_cache_tag": str(getattr(sys.implementation, "cache_tag", "")),
        "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
    }


def _mlx_distribution_version(site: Path) -> str:
    for distribution in metadata.distributions(path=[str(site)]):
        if distribution.metadata.get("Name", "").lower() == "mlx":
            return distribution.version
    return "missing"


def _runtime_fingerprints() -> dict[str, str]:
    paths = (
        FORGE_ROOT / "backend/modules/k_model.py",
        EXTENSION_ROOT / "forge_apple_accelerator/provider.py",
        EXTENSION_ROOT / "forge_apple_accelerator/runtime/mlx_anima_denoiser.py",
        Path(__file__).resolve(),
        FORGE_ROOT / "scripts/apple_mlx/anima_mlx.py",
    )
    return {str(path.relative_to(FORGE_ROOT)): _sha256(path) for path in paths}


def _configured_manifest_path() -> Path:
    configured = os.getenv("FORGE_APPLE_MLX_SYNC_MANIFEST", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MANIFEST_PATH


def _validate(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BridgeUnavailable(f"manifest_unreadable:{type(exc).__name__}:{exc}") from exc

    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("ok") is not True:
        raise BridgeUnavailable("manifest_schema_or_status_mismatch")
    if manifest.get("required_symbol") != _REQUIRED_SYMBOL or manifest.get("symbol_verified") is not True:
        raise BridgeUnavailable("required_symbol_unverified")
    if manifest.get("compatibility") != _runtime_compatibility():
        raise BridgeUnavailable("runtime_compatibility_mismatch")
    if manifest.get("runtime_fingerprints") != _runtime_fingerprints():
        raise BridgeUnavailable("forge_runtime_fingerprint_mismatch")
    if manifest.get("source", {}).get("sha256") != _sha256(SOURCE_PATH):
        raise BridgeUnavailable("source_hash_mismatch")

    mlx_site = Path(os.getenv("FORGE_APPLE_MLX_SITE", "")).expanduser().resolve()
    if not mlx_site.is_dir():
        raise BridgeUnavailable("mlx_site_missing")
    libmlx = mlx_site / "mlx/lib/libmlx.dylib"
    core_module = next((mlx_site / "mlx").glob("core.*.so"), None)
    if core_module is None or not libmlx.is_file():
        raise BridgeUnavailable("mlx_runtime_files_missing")
    expected_mlx = {
        "site": str(mlx_site),
        "version": _mlx_distribution_version(mlx_site),
        "libmlx_file": str(libmlx),
        "libmlx_sha256": _sha256(libmlx),
        "core_module_file": str(core_module),
        "core_module_sha256": _sha256(core_module),
    }
    if expected_mlx["version"] != REQUIRED_MLX or manifest.get("mlx_runtime") != expected_mlx:
        raise BridgeUnavailable("mlx_runtime_fingerprint_mismatch")

    module_path = Path(str(manifest.get("module", {}).get("file", ""))).expanduser().resolve()
    if not module_path.is_file():
        raise BridgeUnavailable("bridge_binary_missing")
    if manifest.get("module", {}).get("sha256") != _sha256(module_path):
        raise BridgeUnavailable("bridge_binary_hash_mismatch")
    return manifest, module_path


def _load() -> tuple[Any, dict[str, Any]]:
    global _BRIDGE, _MANIFEST, _LOAD_ERROR
    if _BRIDGE is not None and _MANIFEST is not None:
        return _BRIDGE, _MANIFEST
    with _LOAD_LOCK:
        if _BRIDGE is not None and _MANIFEST is not None:
            return _BRIDGE, _MANIFEST
        try:
            manifest, module_path = _validate(_configured_manifest_path())
            module_name = str(manifest.get("module", {}).get("name", ""))
            if module_name != "forge_apple_mlx_gil_sync_ext":
                raise BridgeUnavailable("bridge_module_name_mismatch")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise BridgeUnavailable("bridge_module_spec_unavailable")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            info = module.runtime_info()
            if info.get("mlx_version") != REQUIRED_MLX:
                raise BridgeUnavailable("loaded_bridge_mlx_version_mismatch")
            if info.get("gil_policy") != "released_only_around_mlx_core_synchronize":
                raise BridgeUnavailable("loaded_bridge_gil_policy_mismatch")
            _BRIDGE = module
            _MANIFEST = manifest
            _LOAD_ERROR = ""
            return module, manifest
        except Exception as exc:
            _LOAD_ERROR = f"{type(exc).__name__}:{exc}"
            raise BridgeUnavailable(_LOAD_ERROR) from exc


def selected_backend() -> str:
    return os.getenv("FORGE_APPLE_MLX_SYNC_BACKEND", BACKEND_PYTHON).strip().lower()


def synchronize(mx: Any) -> None:
    backend = selected_backend()
    if backend == BACKEND_PYTHON:
        mx.synchronize()
        return
    if backend != BACKEND_GIL_SAFE_NATIVE:
        raise BridgeUnavailable(f"unsupported_sync_backend:{backend}")
    bridge, _manifest = _load()
    bridge.synchronize()


def status() -> dict[str, Any]:
    return {
        "selected_backend": selected_backend(),
        "loaded": _BRIDGE is not None,
        "load_error": _LOAD_ERROR,
        "manifest": str(_configured_manifest_path()),
        "required_mlx": REQUIRED_MLX,
        "required_symbol": _REQUIRED_SYMBOL,
        "gil_policy": "released_only_around_mlx_core_synchronize",
    }
