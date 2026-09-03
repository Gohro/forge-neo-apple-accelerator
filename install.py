"""Validate or build the extension-owned exact-runtime MPSGraph bridge."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parent
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, validate_manifest


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("Forge Apple Accelerator: non-Apple-Silicon host; native bridge disabled.")
        return 0

    try:
        import torch
    except Exception as exc:
        print(f"Forge Apple Accelerator: cannot import Torch; native bridge disabled: {exc}")
        return 0

    valid, reason, _manifest, _module_path = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
    if valid:
        print(f"Forge Apple Accelerator: native bridge ready for Torch {torch.__version__}.")
        return 0

    command = [
        sys.executable,
        "-m",
        "forge_apple_accelerator.build_bridge",
        "--build-dir",
        str(DEFAULT_MANIFEST_PATH.parent),
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(EXTENSION_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    try:
        subprocess.run(command, cwd=EXTENSION_ROOT, env=environment, check=True)
    except Exception as exc:
        print(
            "Forge Apple Accelerator: bridge build failed; guarded fallback will remain active "
            f"({reason}): {exc}"
        )
        return 0

    valid, reason, _manifest, _module_path = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
    if valid:
        print(f"Forge Apple Accelerator: built native bridge for Torch {torch.__version__}.")
    else:
        print(f"Forge Apple Accelerator: bridge validation failed ({reason}); guarded fallback will remain active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
