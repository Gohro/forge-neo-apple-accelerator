<div align="center">

<h1>Forge Neo Apple Accelerator</h1>
<p><strong>Normal Anima generation in 23.9% less time. Hi-Res Fix in 35.4% less time.</strong></p>
<p>
  <img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-native-black?logo=apple">
  <img alt="Forge Neo" src="https://img.shields.io/badge/Forge%20Neo-neo%20branch-6f42c1">
  <img alt="Normal Anima generation" src="https://img.shields.io/badge/normal%20Anima-23.9%25%20less%20time-0a7f5a">
  <img alt="Anima Hi-Res Fix" src="https://img.shields.io/badge/Anima%20Hi--Res-35.4%25%20less%20time-2563eb">
  <img alt="Release 0.2" src="https://img.shields.io/badge/release-v0.2-2563eb">
</p>

</div>

Keep the Forge interface, prompts, samplers, checkpoints, and workflows you
already use. Install the extension, restart Forge, and its qualified Apple
routes activate automatically. There are no separate nightly or performance
switches.

## Results at a glance

Measured on the M5 Pro development Mac with fresh Forge processes, identical
prompts and seeds, counterbalanced run order, and nominal thermal checks:

| Accelerated workflow | Add-on disabled | Add-on enabled | Measured improvement |
| --- | ---: | ---: | ---: |
| **Normal Anima generation** — 1024×1024, 10 Euler a steps | 22.74s | 17.30s | **23.9% less time / 1.31×** |
| **Complete Anima Hi-Res Fix** — 1024→1536, 10+10 steps | 134.49s | 86.88s | **35.4% less time / 1.55×** |
| SwinIR stage inside that Hi-Res request | 42.57s | 12.40s | **70.9% less time / 3.43×** |

[Normal-generation evidence](evidence/anima-v02-release-matrix-20260904.json)
· [Hi-Res evidence](evidence/hires-release-matrix-20260904.json)

> [!IMPORTANT]
> Normal text-to-image acceleration is currently **Anima-only**. SDXL models
> such as Nova Comic XL do not receive a base-generation speedup yet.

## What is accelerated

| Workflow | Support | Released behavior |
| --- | --- | --- |
| Normal text-to-image | Anima | Metal RoPE + guarded MPSGraph attention |
| Hi-Res Fix | Anima + `SwinIR_4x.pth` | Stable Torch 2.14 VAE + BF16/compiled SwinIR + GPU compositing |
| Normal text-to-image | SDXL / Illustrious / Nova XL | No denoiser speedup yet |
| Tiled / Ultimate upscale | — | Not released yet |

Unsupported models, shapes, and dtypes fall back to Forge. See
[MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the exact boundary.

## Benchmark details

All public comparisons use fresh Forge processes, identical prompts and seeds,
and explicit thermal checks on the M5 Pro development Mac.

### Normal Anima generation

The packaged v0.2 lifecycle was tested at 1024×1024 and 10 Euler a steps in
`accelerated → vanilla → vanilla → accelerated` order. All thermal boundaries
were nominal.

| Add-on disabled median | Add-on enabled median | Improvement |
| ---: | ---: | ---: |
| 22.74s | 17.30s | **23.9% less / 1.31×** |

Same-route repeats were pixel-exact. Cross-route SSIM was 0.93084, with the same
composition, detail level, and overall visual quality under full review.

[Machine-readable v0.2 normal-generation matrix](evidence/anima-v02-release-matrix-20260904.json)

### Hi-Res Fix: installed add-on vs disabled add-on

Counterbalanced `accelerated → vanilla → vanilla → accelerated` test,
1024→1536, 10+10 Euler a steps, `SwinIR_4x.pth`:

| Measurement | Add-on disabled | Add-on enabled | Improvement |
| --- | ---: | ---: | ---: |
| Complete request median | 134.49s | 86.88s | **35.4% less / 1.55×** |
| SwinIR median | 42.57s | 12.40s | **70.9% less / 3.43×** |

The two vanilla outputs were pixel-exact to each other, and the two accelerated
outputs were pixel-exact to each other. Cross-route SSIM was 0.91819; visual
review found the same subject, composition, detail level, and overall quality,
with minor line/highlight placement drift. This is visual parity, not strict
same-seed identity across runtimes.

The very first Hi-Res request on a clean install builds the SwinIR graph cache:
112.47s total and 35.26s in SwinIR in the measured run. After restart, the
cached samples were 91.78s and 81.98s. Clean-cache and cached outputs were
pixel-exact.

On the 48 GB test Mac, a 1024 source uses one full-image MPS SwinIR call. It is
normal for the old tile progress bar to disappear: no CPU tile loop runs on
that path. Lower-memory Macs automatically retain a conservative tiled route.

[Machine-readable Hi-Res release matrix](evidence/hires-release-matrix-20260904.json)

Full timings, route proof, and caveats are in [BENCHMARKS.md](BENCHMARKS.md).

## Install

1. Open **Extensions → Install from URL** in Forge Neo.
2. Paste `https://github.com/Gohro/forge-neo-apple-accelerator` and install.
3. In **Installed**, choose **Apply and restart UI**, then fully stop and restart
   the Forge backend once.

Installation downloads the pinned official stable `torch 2.14.0` and
`torchvision 0.29.0` Apple wheels into an extension-owned overlay, verifies
their SHA-256 hashes, and builds an exact-runtime native bridge. It does not
replace the Torch packages in Forge's virtual environment.

While the extension is enabled, a small startup bootstrap selects that verified
overlay before Forge imports Torch. Disable the extension in **Installed** and
restart to return to Forge's original runtime and behavior.

### Requirements

- Apple Silicon Mac (`arm64`)
- Forge Classic / Forge Neo [`neo` branch](https://github.com/Haoming02/sd-webui-forge-classic)
- Xcode Command Line Tools
- Forge's current Python 3.13 environment
- Internet access during the first install for the pinned PyTorch wheels

Tested Forge commit: `e91c293170e4451fc27490827f9aa62ccc77e6dc`.

## Verify

Open **Settings → Apple Accelerator** for a read-only runtime/route status table,
or query:

```text
GET /internal/forge-apple-accelerator/status
```

Native self-test:

```bash
PYTHONPATH=extensions/forge_apple_accelerator \
  venv/bin/python -m forge_apple_accelerator.self_test
```

## Current limits

- SDXL base generation is not accelerated in v0.2.
- Released Hi-Res measurements cover Anima with the exact `SwinIR_4x.pth`
  checkpoint on the 48 GB development Mac; lower-memory Macs use a conservative
  512 tile automatically and need a separate published hardware matrix.
- The first use of a new SwinIR input shape builds a persistent graph cache.
- Ultimate SD Upscale is not yet claimed as accelerated.

## Project docs

- [Model support](MODEL_SUPPORT.md)
- [Benchmark evidence](BENCHMARKS.md)
- [Changelog](CHANGELOG.md)
- [License and notices](LICENSE)

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
