"""Select the extension-owned stable runtime before Forge imports Torch.

This module is invoked by a one-line venv ``.pth`` file written by install.py.
It intentionally uses only the Python standard library. Disabling this
extension in Forge's config makes the bootstrap a no-op on the next restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parent
FORGE_ROOT = EXTENSION_ROOT.parents[1]
EXTENSION_NAME = EXTENSION_ROOT.name
RUNTIME_MANIFEST = EXTENSION_ROOT / "runtime_overlay_manifest.json"
RUNTIME_OVERLAY = EXTENSION_ROOT / "runtime_overlay"
RUNTIME_STAMP = RUNTIME_OVERLAY / "forge_apple_runtime.json"
STABLE_BRIDGE_MANIFEST = EXTENSION_ROOT / "native_build/mpsgraph_attention_interop-stable/manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings_path() -> Path:
    for index, value in enumerate(sys.argv):
        if value == "--ui-settings-file" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser().resolve()
        if value.startswith("--ui-settings-file="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return FORGE_ROOT / "config.json"


def _extension_enabled() -> bool:
    if {"--disable-all-extensions", "--disable-extra-extensions"}.intersection(sys.argv):
        return False
    try:
        settings = json.loads(_settings_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        settings = {}
    except Exception:
        return False
    if settings.get("disable_all_extensions", "none") != "none":
        return False
    return EXTENSION_NAME not in set(settings.get("disabled_extensions", []))


def activate() -> bool:
    if os.getenv("FORGE_APPLE_DISABLE_EARLY_BOOTSTRAP") == "1":
        return False
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    if not _extension_enabled() or "torch" in sys.modules:
        return False
    try:
        manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
        stamp = json.loads(RUNTIME_STAMP.read_text(encoding="utf-8"))
        if stamp.get("runtime_id") != manifest.get("runtime_id"):
            return False
        if stamp.get("manifest_sha256") != _sha256(RUNTIME_MANIFEST):
            return False
        if not STABLE_BRIDGE_MANIFEST.is_file():
            return False
    except Exception:
        return False

    sys.path.insert(0, str(RUNTIME_OVERLAY))
    os.environ["FORGE_APPLE_STABLE_RUNTIME_ACTIVE"] = "1"
    os.environ["FORGE_APPLE_MPSGRAPH_ATTENTION_MANIFEST"] = str(STABLE_BRIDGE_MANIFEST)
    os.environ["FORGE_APPLE_MPSGRAPH_EXTENSION_MANIFEST"] = str(STABLE_BRIDGE_MANIFEST)
    cache = EXTENSION_ROOT / "runtime_cache/torchinductor"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))
    return True


activate()
