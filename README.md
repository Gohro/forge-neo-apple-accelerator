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

## Default installation: measured gains

These numbers are for the normal extension install using stable Forge Torch
2.12.1 and `visual-fast`. They do **not** include the optional nightly runtime.

Fresh-process ABBA test on the M5 Pro development Mac, 2026-09-03. Every pair
used the same 1024×1024 Anima checkpoint, prompt, seed, GPU RNG, sampler,
scheduler, and settings, with two samples per route:

| Steps | Vanilla samples / median | Default add-on samples / median | Median reduction |
| ---: | ---: | ---: | ---: |
| 10 | 29.40, 37.23s / **33.31s** | 17.67, 22.24s / **19.96s** | **40.1%** |
| 15 | 52.70, 42.73s / **47.72s** | 25.39, 25.27s / **25.33s** | **46.9%** |
| 30 | 88.30, 83.10s / **85.70s** | 48.76, 48.74s / **48.75s** | **43.1%** |

Same-route image repeats were pixel-exact. Vanilla and `visual-fast` images
were visually comparable but not pixel-exact; `visual-fast` is a visual-parity
mode, not a strict same-seed reproduction mode.

See the [machine-readable release matrix](evidence/anima-stable-release-matrix-20260903.json)
for every retained timing, route check, thermal event, and image comparison.

Additional qualified component results:

- Hi-Res Anima sampler loop: approximately **218s → 144s**.
- SwinIR CPU-to-GPU compositor change: **60.687s → 43.111s** (**29.0%**).

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

The normal installation uses Forge's existing Torch runtime. It does not
download or enable the optional nightly runtime.

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

This is a second acceleration layer, not part of the default install. Historical
tests measured another 13–16% reduction versus the already accelerated stable
route. Only the 10-step result has a two-pair ABBA qualification; the 15- and
30-step results are single cooled pairs. They are therefore not combined with
the default-install table above.

The word “exact” means that its Anima denoiser trajectory matches the stable
accelerated route; it does not mean that either route is pixel-identical to
vanilla Forge.

Install the pinned overlay once:

```bash
venv/bin/python extensions/forge_apple_accelerator/install_runtime.py
```

Enable it by fully stopping Forge and launching through:

```bash
extensions/forge_apple_accelerator/launch_apple_accelerated.sh
```

It is pinned to `torch 2.15.0.dev20260901` and
`torchvision 0.30.0.dev20260901`. It shadows Torch only for that launch and
does not modify Forge's virtual environment.

Disable it by fully stopping that process and starting Forge normally through
your usual launcher. Version 0.1 does not yet expose a safe UI runtime toggle:
Torch must be selected before Forge imports it. A restart-aware add-on launcher
and settings control require a separate lifecycle qualification before they can
replace the dedicated script.

No combined vanilla-to-nightly percentage is claimed until a fresh three-route
vanilla / stable / nightly matrix passes timing, restart, and visual gates.

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

To disable the stable add-on routes, select `off` and fully restart, or disable
the extension in Forge's **Installed** tab. To disable the optional nightly
runtime, quit its process and return to your normal Forge launcher.

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
