#!/usr/bin/env python3
"""Build and fingerprint the exact-version MLX GIL-safe sync bridge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from forge_apple_accelerator.runtime import mlx_sync_bridge


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _distribution_version(site: Path, name: str) -> str:
    for distribution in metadata.distributions(path=[str(site)]):
        if distribution.metadata.get("Name", "").lower() == name.lower():
            return distribution.version
    return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-site", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=mlx_sync_bridge.DEFAULT_MANIFEST_PATH.parent)
    args = parser.parse_args()

    mlx_site = args.mlx_site.expanduser().resolve()
    build_dir = args.build_dir.expanduser().resolve()
    include_dir = mlx_site / "mlx/include"
    lib_dir = mlx_site / "mlx/lib"
    libmlx = lib_dir / "libmlx.dylib"
    core_module = next((mlx_site / "mlx").glob("core.*.so"), None)
    if _distribution_version(mlx_site, "mlx") != mlx_sync_bridge.REQUIRED_MLX:
        raise RuntimeError(f"exact MLX {mlx_sync_bridge.REQUIRED_MLX} installation required")
    if _distribution_version(mlx_site, "mlx-metal") != mlx_sync_bridge.REQUIRED_MLX:
        raise RuntimeError(f"exact mlx-metal {mlx_sync_bridge.REQUIRED_MLX} installation required")
    if core_module is None or not libmlx.is_file():
        raise RuntimeError("incomplete MLX runtime")

    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    module_path = build_dir / f"forge_apple_mlx_gil_sync_ext{suffix}"
    source = mlx_sync_bridge.SOURCE_PATH
    compiler = subprocess.run(
        ["xcrun", "--find", "clang++"], check=True, capture_output=True, text=True
    ).stdout.strip()
    sdk_path = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], check=True, capture_output=True, text=True
    ).stdout.strip()
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fvisibility=hidden",
        "-isysroot",
        sdk_path,
        "-bundle",
        "-undefined",
        "dynamic_lookup",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{include_dir}",
        str(source),
        f"-L{lib_dir}",
        "-lmlx",
        f"-Wl,-rpath,{lib_dir}",
        "-o",
        str(module_path),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    build_seconds = time.perf_counter() - started

    nm_output = subprocess.run(["nm", "-gU", str(libmlx)], check=True, capture_output=True, text=True).stdout
    symbol_verified = mlx_sync_bridge._REQUIRED_SYMBOL in nm_output
    if not symbol_verified:
        raise RuntimeError("required MLX synchronize symbol is not exported")
    dependencies = subprocess.run(["otool", "-L", str(module_path)], check=True, capture_output=True, text=True).stdout
    rpaths = subprocess.run(["otool", "-l", str(module_path)], check=True, capture_output=True, text=True).stdout

    os.environ["FORGE_APPLE_MLX_SITE"] = str(mlx_site)
    manifest = {
        "schema_version": mlx_sync_bridge.SCHEMA_VERSION,
        "ok": True,
        "mechanism": "A_release_python_gil_only_around_native_mlx_synchronize",
        "required_symbol": mlx_sync_bridge._REQUIRED_SYMBOL,
        "symbol_verified": symbol_verified,
        "source": {"file": str(source), "sha256": _sha256(source)},
        "module": {
            "name": "forge_apple_mlx_gil_sync_ext",
            "file": str(module_path),
            "sha256": _sha256(module_path),
        },
        "compatibility": mlx_sync_bridge._runtime_compatibility(),
        "runtime_fingerprints": mlx_sync_bridge._runtime_fingerprints(),
        "mlx_runtime": {
            "site": str(mlx_site),
            "version": _distribution_version(mlx_site, "mlx"),
            "libmlx_file": str(libmlx),
            "libmlx_sha256": _sha256(libmlx),
            "core_module_file": str(core_module),
            "core_module_sha256": _sha256(core_module),
        },
        "mlx_metal_version": _distribution_version(mlx_site, "mlx-metal"),
        "build": {
            "compiler": compiler,
            "sdk_path": sdk_path,
            "command": command,
            "seconds": build_seconds,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
            "dependencies": dependencies.splitlines(),
            "load_commands_sha256": hashlib.sha256(rpaths.encode("utf-8")).hexdigest(),
            "built_at_unix": time.time(),
        },
        "gate": {
            "default_enabled": False,
            "exact_version_only": True,
            "no_python_api_or_refcount_while_gil_released": True,
            "global_provider_lock_retained": True,
        },
    }
    manifest_path = build_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    valid_manifest, valid_module = mlx_sync_bridge._validate(manifest_path)
    spec = importlib.util.spec_from_file_location(valid_manifest["module"]["name"], valid_module)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import built bridge")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime_info = module.runtime_info()
    if runtime_info.get("mlx_version") != mlx_sync_bridge.REQUIRED_MLX:
        raise RuntimeError(f"built bridge loaded wrong MLX runtime: {runtime_info}")
    print(json.dumps({"manifest": str(manifest_path), "module": str(module_path), "runtime_info": runtime_info}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
