#!/usr/bin/env bash

set -euo pipefail

extension_root="$(cd "$(dirname "$0")" && pwd)"
forge_root="$(cd "$extension_root/../.." && pwd)"
forge_python="$forge_root/venv/bin/python"
overlay="$extension_root/runtime_overlay"
stable_build="$extension_root/native_build/mpsgraph_attention_interop-stable"
stable_manifest="$stable_build/manifest.json"

if [ ! -x "$forge_python" ]; then
    echo "Forge Neo Python was not found at: $forge_python" >&2
    exit 1
fi

if ! "$forge_python" "$extension_root/install_runtime.py" --check >/dev/null; then
    "$forge_python" "$extension_root/install_runtime.py" --replace
fi

export PYTHONPATH="$overlay:$extension_root${PYTHONPATH:+:$PYTHONPATH}"
export FORGE_APPLE_ACCELERATOR_MODE="inherit"
export FORGE_APPLE_BACKEND="off"
export FORGE_APPLE_DENOISER_BACKEND="off"
export FORGE_APPLE_METAL_KERNELS="rope"
export FORGE_APPLE_ATTENTION_BACKEND="torch-sdpa-self-mpsgraph-cross"
export FORGE_APPLE_ATTENTION_CROSS="1"
export FORGE_APPLE_ANIMA_COMPAT_BACKEND="off"
export FORGE_APPLE_UPSCALER_GPU_COMPOSITE="1"
export FORGE_APPLE_SWINIR_BF16="1"
export FORGE_APPLE_SWINIR_COMPILE="1"
export FORGE_APPLE_UPSCALER_TILE="768"
export FORGE_APPLE_MPSGRAPH_ATTENTION_MANIFEST="$stable_manifest"
export FORGE_APPLE_MPSGRAPH_EXTENSION_MANIFEST="$stable_manifest"
export TORCHINDUCTOR_CACHE_DIR="$extension_root/runtime_cache/torchinductor"

mkdir -p "$stable_build" "$TORCHINDUCTOR_CACHE_DIR"
"$forge_python" -m forge_apple_accelerator.build_bridge --build-dir "$stable_build"

cd "$forge_root"
export PYTHON="$forge_python"
export SKIP_VENV="1"
exec "$forge_root/webui.sh" --skip-prepare-environment --skip-install "$@"
