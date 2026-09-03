#!/usr/bin/env python3
"""Build the extension-owned exact-version MPSGraph/PyTorch bridge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.utils.cpp_extension as cpp_extension

from forge_apple_accelerator import __version__
from forge_apple_accelerator.bridge_manifest import (
    DEFAULT_MANIFEST_PATH,
    SOURCE_PATH,
    extension_runtime_fingerprint,
    forge_adapter_fingerprint,
    runtime_compatibility,
    sha256_file,
)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def patch_torch_lib_path_for_spaces() -> str:
    torch_lib = Path(cpp_extension.TORCH_LIB_PATH)
    if " " not in str(torch_lib):
        return str(torch_lib)
    symlink_path = Path("/tmp/forge_apple_accelerator_torch_lib")
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(torch_lib, target_is_directory=True)
    cpp_extension.TORCH_LIB_PATH = str(symlink_path)
    return str(symlink_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(DEFAULT_MANIFEST_PATH.parent))
    parser.add_argument("--source", default=str(SOURCE_PATH))
    parser.add_argument("--name", default="forge_apple_mpsgraph_ext")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = build_dir / "manifest.json"

    started = time.perf_counter()
    patched_torch_lib_path = patch_torch_lib_path_for_spaces()
    module = cpp_extension.load(
        name=args.name,
        sources=[str(source_path)],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_ldflags=[
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "MetalPerformanceShaders",
            "-framework",
            "MetalPerformanceShadersGraph",
        ],
        verbose=args.verbose,
        is_python_module=True,
    )
    elapsed = time.perf_counter() - started

    runtime_probe = module.probe_runtime()
    tensor_probe: dict[str, Any] = {}
    if torch.backends.mps.is_available():
        sample = torch.empty((2, 4096, 16, 128), device="mps", dtype=torch.bfloat16)
        tensor_probe = module.probe_tensor(sample)
        torch.mps.synchronize()

    module_path = Path(module.__file__).resolve()
    manifest = {
        "schema_version": 2,
        "ok": True,
        "extension_version": __version__,
        "module_name": args.name,
        "module_file": str(module_path),
        "module": {
            "name": args.name,
            "file": str(module_path),
            "sha256": sha256_file(module_path),
        },
        "source": {
            "file": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "build_dir": str(build_dir),
        "compatibility": runtime_compatibility(torch),
        "forge_adapter": forge_adapter_fingerprint(),
        "extension_runtime": extension_runtime_fingerprint(),
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "torch_lib_path": patched_torch_lib_path,
        "built_at_unix": time.time(),
        "build_elapsed_s": elapsed,
        "runtime_probe": runtime_probe,
        "tensor_probe": tensor_probe,
        "gate": {
            "requires_exact_runtime_fingerprint": True,
            "uses_internal_mps_headers": True,
            "default_forge_route_enabled": False,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "module_file": str(module_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
