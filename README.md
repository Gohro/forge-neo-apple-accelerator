<div align="center">

<h1>Forge Neo Apple Accelerator</h1>
<p><strong>Faster Anima generation in Forge Neo on Apple Silicon.</strong></p>
<p>
  <img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-native-black?logo=apple">
  <img alt="Forge Neo" src="https://img.shields.io/badge/Forge%20Neo-neo%20branch-6f42c1">
  <img alt="Model support: Anima only" src="https://img.shields.io/badge/base%20generation-Anima%20only-0a7f5a">
  <img alt="v0.1 developer preview" src="https://img.shields.io/badge/release-v0.1%20developer%20preview-d97706">
</p>

</div>

Keep the Forge interface, prompts, samplers, checkpoints, and workflows you
already use. This extension routes qualified Anima operations through guarded
Metal and MPSGraph paths, and improves the SwinIR upscaler path on compatible
Apple Silicon Macs.

> [!IMPORTANT]
> Version 0.1 is a developer preview. **Normal text-to-image acceleration is
> currently Anima-only. SDXL models such as Nova Comic XL do not receive a
> base-generation speedup yet.**

## At a glance

| Workflow | Model support | Status |
| --- | --- | --- |
| Normal text-to-image | **Anima** | Accelerated in `visual-fast` |
| Normal text-to-image | SDXL / Illustrious / Nova XL | Not accelerated yet |
| Hi-Res second denoiser | Anima | Component-qualified; full-request gate pending |
| SwinIR upscale stage | Model-agnostic | GPU compositor qualified |
| Tiled / Ultimate upscale | — | Planned |

See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the precise support boundary.

## Measured gains

Same M5 Pro development Mac, 1024×1024, same Anima checkpoint, prompt, seed,
sampler, scheduler, and settings:

| Steps | Vanilla Forge | Accelerator | Time reduction |
| ---: | ---: | ---: | ---: |
| 10 | 38.47s | 19.74s | **48.7%** |
| 15 | 55.95s | 28.39s | **49.3%** |
| 30 | 120.84s | 68.83s | **43.0%** |

Additional qualified component results:

- Hi-Res Anima sampler loop: approximately **218s → 144s**.
- SwinIR CPU-to-GPU compositor change: **60.687s → 43.111s** (**29.0%**).
- Optional exact-nightly runtime: another **13–16%** over the already
  accelerated normal-generation route.

These are scoped results, not a promise that every model or complete Hi-Res
request improves by the same percentage. Full settings, parity evidence, and
thermal protocol are in [BENCHMARKS.md](BENCHMARKS.md).

## Install

1. In Forge Neo, open **Extensions → Install from URL**.
2. Paste `https://github.com/Gohro/forge-neo-apple-accelerator` and install.
3. Open **Installed**, choose **Apply and restart UI**, then fully restart the
   Forge backend.

After restart, open **Settings → Apple Accelerator**. The default
`visual-fast` mode enables the qualified routes.

### Requirements

- Apple Silicon Mac (`arm64`)
- Forge Classic / Forge Neo [`neo` branch](https://github.com/Haoming02/sd-webui-forge-classic)
- Xcode Command Line Tools
- Forge's current Python 3.13 environment

Tested Forge commit: `e91c293170e4451fc27490827f9aa62ccc77e6dc`.

## Modes

| Mode | Behavior |
| --- | --- |
| `off` | Normal Forge / PyTorch MPS |
| `strict` | Exact promoted Metal RoPE route only |
| `visual-fast` | Qualified Anima Metal/MPSGraph routes plus Apple SwinIR settings |
| `inherit` | Preserve settings supplied by an external launcher |

Unsupported devices, dtypes, shapes, and model families fall back to Forge.
Mode or tile changes require a full backend restart.

## Optional exact-nightly preview

The opt-in runtime adds 13–16% over the accelerated Anima route without
modifying Forge's main virtual environment:

```bash
venv/bin/python extensions/forge_apple_accelerator/install_runtime.py
extensions/forge_apple_accelerator/launch_apple_accelerated.sh
```

It is pinned to `torch 2.15.0.dev20260901` and
`torchvision 0.30.0.dev20260901`. Normal installation does not download it.

## Verify or disable

Runtime status:

```text
GET /internal/forge-apple-accelerator/status
```

Native self-test:

```bash
PYTHONPATH=extensions/forge_apple_accelerator \
  venv/bin/python -m forge_apple_accelerator.self_test
```

To disable the add-on, select `off` or disable it in Forge's **Installed** tab
and restart. The optional nightly runtime is active only when launched through
its dedicated script.

## Current limits

- SDXL base generation is not accelerated in v0.1.
- Complete cooled Hi-Res qualification is still pending; current Hi-Res and
  SwinIR numbers are component-scoped.
- Ultimate SD Upscale is not yet claimed as accelerated.
- v0.1 targets the tested Forge Neo revision and remains a developer preview.

## Project docs

- [Model support](MODEL_SUPPORT.md)
- [Benchmark evidence](BENCHMARKS.md)
- [Changelog](CHANGELOG.md)
- [License and notices](LICENSE)

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
