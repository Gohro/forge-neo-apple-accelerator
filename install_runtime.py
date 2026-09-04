#!/usr/bin/env python3
"""Install or inspect the extension-owned stable Apple runtime overlay.

The overlay shadows only Torch and TorchVision.  It never modifies Forge Neo's
venv. The extension preload selects it before Torch imports, and disabling the
extension returns Forge to its original runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "runtime_overlay_manifest.json"
DEFAULT_OVERLAY = ROOT / "runtime_overlay"
STAMP_NAME = "forge_apple_runtime.json"


def load_manifest() -> dict:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported runtime manifest schema")
    return payload


def check_host(manifest: dict) -> None:
    required_python = manifest["python"]
    if sys.version_info[:2] != (required_python["major"], required_python["minor"]):
        raise RuntimeError(
            f"runtime requires Python {required_python['major']}.{required_python['minor']}; "
            f"found {platform.python_version()}"
        )
    required_platform = manifest["platform"]
    if platform.system() != required_platform["system"] or platform.machine() != required_platform["machine"]:
        raise RuntimeError(
            f"runtime requires {required_platform['system']} {required_platform['machine']}; "
            f"found {platform.system()} {platform.machine()}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_status(overlay: Path, manifest: dict) -> tuple[bool, str]:
    stamp_path = overlay / STAMP_NAME
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"stamp_unavailable:{type(exc).__name__}"
    if stamp.get("runtime_id") != manifest.get("runtime_id"):
        return False, "runtime_id_mismatch"
    if stamp.get("manifest_sha256") != sha256(MANIFEST_PATH):
        return False, "manifest_hash_mismatch"
    for package in manifest["packages"]:
        distribution = overlay / f"{package['name']}-{package['version']}.dist-info"
        if not distribution.is_dir():
            return False, f"missing_distribution:{distribution.name}"
    return True, "ok"


def download(package: dict, destination: Path) -> Path:
    target = destination / package["filename"]
    print(f"Downloading {package['name']} {package['version']}...")
    with urllib.request.urlopen(package["url"]) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual_size = target.stat().st_size
    if actual_size != package["size"]:
        raise RuntimeError(f"size mismatch for {package['filename']}: {actual_size}")
    actual_hash = sha256(target)
    if actual_hash != package["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {package['filename']}")
    return target


def verify_import(overlay: Path, manifest: dict) -> None:
    expected = {package["name"]: package["version"] for package in manifest["packages"]}
    code = (
        "import json, torch, torchvision; "
        "print(json.dumps({'torch': getattr(torch, '__long_version__', torch.__version__), "
        "'torchvision': torchvision.__version__, 'torch_file': torch.__file__}))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(overlay)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not str(payload["torch"]).startswith(expected["torch"]):
        raise RuntimeError(f"unexpected Torch import: {payload}")
    if not str(payload["torchvision"]).startswith(expected["torchvision"]):
        raise RuntimeError(f"unexpected TorchVision import: {payload}")
    if not str(Path(payload["torch_file"]).resolve()).startswith(str(overlay.resolve())):
        raise RuntimeError(f"Torch did not import from overlay: {payload['torch_file']}")


def install(overlay: Path, manifest: dict, *, replace: bool) -> None:
    ready, reason = installed_status(overlay, manifest)
    if ready:
        print(f"Stable Apple runtime already ready: {overlay}")
        return
    if overlay.exists() and any(overlay.iterdir()):
        if not replace:
            raise RuntimeError(
                f"overlay exists but is not valid ({reason}); rerun with --replace to rebuild only {overlay}"
            )
        shutil.rmtree(overlay)
    overlay.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="forge-apple-runtime-") as temporary:
        download_dir = Path(temporary)
        wheels = [download(package, download_dir) for package in manifest["packages"]]
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--upgrade",
            "--target",
            str(overlay),
            *(str(wheel) for wheel in wheels),
        ]
        subprocess.run(command, check=True)

    stamp = {
        "schema_version": 1,
        "runtime_id": manifest["runtime_id"],
        "manifest_sha256": sha256(MANIFEST_PATH),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": {package["name"]: package["version"] for package in manifest["packages"]},
    }
    (overlay / STAMP_NAME).write_text(json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8")
    verify_import(overlay, manifest)
    print(f"Stable Apple runtime ready: {overlay}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--check", action="store_true", help="check without downloading or changing files")
    parser.add_argument("--replace", action="store_true", help="replace only an invalid extension-owned overlay")
    args = parser.parse_args()

    manifest = load_manifest()
    check_host(manifest)
    overlay = args.overlay.expanduser().resolve()
    ready, reason = installed_status(overlay, manifest)
    if args.check:
        print(json.dumps({"ready": ready, "reason": reason, "overlay": str(overlay)}, indent=2))
        return 0 if ready else 1
    install(overlay, manifest, replace=args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
