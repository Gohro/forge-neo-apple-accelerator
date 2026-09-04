"""Install the stable Apple runtime and its exact-runtime native bridges."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import sysconfig
import json
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parent
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from forge_apple_accelerator.bridge_manifest import DEFAULT_MANIFEST_PATH, validate_manifest
from install_runtime import DEFAULT_OVERLAY, check_host, install as install_runtime, load_manifest


STABLE_BUILD_DIR = EXTENSION_ROOT / "native_build/mpsgraph_attention_interop-stable"
BOOTSTRAP_PATH = EXTENSION_ROOT / "early_bootstrap.py"
BOOTSTRAP_PTH_NAME = "forge_apple_accelerator_bootstrap.pth"


def build_bridge(build_dir: Path, *, overlay: Path | None = None) -> bool:
    command = [
        sys.executable,
        "-m",
        "forge_apple_accelerator.build_bridge",
        "--build-dir",
        str(build_dir),
    ]
    environment = os.environ.copy()
    if overlay is None:
        environment["FORGE_APPLE_DISABLE_EARLY_BOOTSTRAP"] = "1"
    else:
        environment.pop("FORGE_APPLE_DISABLE_EARLY_BOOTSTRAP", None)
    roots = [str(EXTENSION_ROOT)]
    if overlay is not None:
        roots.insert(0, str(overlay))
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [*roots, *([existing_pythonpath] if existing_pythonpath else [])]
    )
    try:
        subprocess.run(command, cwd=EXTENSION_ROOT, env=environment, check=True)
    except Exception as exc:
        print(f"Forge Apple Accelerator: bridge build failed for {build_dir.name}: {exc}")
        return False
    return True


def install_early_bootstrap() -> Path:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    target = site_packages / BOOTSTRAP_PTH_NAME
    quoted_bootstrap = json.dumps(str(BOOTSTRAP_PATH.resolve()))
    line = (
        f"import os,runpy; os.path.isfile({quoted_bootstrap}) and "
        f"runpy.run_path({quoted_bootstrap}, run_name='_forge_apple_early_bootstrap')\n"
    )
    target.write_text(line, encoding="utf-8")
    return target


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("Forge Apple Accelerator: non-Apple-Silicon host; native bridge disabled.")
        return 0

    try:
        import torch
    except Exception as exc:
        print(f"Forge Apple Accelerator: cannot import Torch; native bridge disabled: {exc}")
        return 0

    torch_file = Path(torch.__file__).resolve()
    running_from_overlay = DEFAULT_OVERLAY.resolve() in torch_file.parents
    if running_from_overlay:
        print("Forge Apple Accelerator: stable runtime is already active; preserving the base-runtime bridge.")
    else:
        valid, reason, _manifest, _module_path = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
        if valid:
            print(f"Forge Apple Accelerator: native bridge ready for Torch {torch.__version__}.")
        else:
            build_bridge(DEFAULT_MANIFEST_PATH.parent)
            valid, reason, _manifest, _module_path = validate_manifest(DEFAULT_MANIFEST_PATH, torch_module=torch)
            if valid:
                print(f"Forge Apple Accelerator: built native bridge for Torch {torch.__version__}.")
            else:
                print(f"Forge Apple Accelerator: bridge validation failed ({reason}); guarded fallback will remain active.")

    try:
        runtime_manifest = load_manifest()
        check_host(runtime_manifest)
        install_runtime(DEFAULT_OVERLAY, runtime_manifest, replace=True)
    except Exception as exc:
        print(
            "Forge Apple Accelerator: stable Torch runtime installation failed; "
            f"the base-runtime acceleration remains available ({type(exc).__name__}: {exc})."
        )
        return 0

    if build_bridge(STABLE_BUILD_DIR, overlay=DEFAULT_OVERLAY):
        try:
            bootstrap = install_early_bootstrap()
        except Exception as exc:
            print(
                "Forge Apple Accelerator: stable runtime is ready, but automatic startup activation failed "
                f"({type(exc).__name__}: {exc}). Use launch_apple_accelerated.sh."
            )
            return 0
        print(
            "Forge Apple Accelerator: stable Torch 2.14 runtime, native bridge, and "
            f"automatic startup bootstrap are ready ({bootstrap})."
        )
    else:
        print("Forge Apple Accelerator: stable runtime is ready, but its native bridge will use guarded fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
