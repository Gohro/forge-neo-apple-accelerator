# Forge Neo Apple Accelerator

An installable Apple Silicon acceleration add-on for the `neo` branch of
[Forge Classic / Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic).
It keeps Forge's prompts, samplers, checkpoints, UI, and feature set, while
routing a small set of measured Anima and SwinIR operations through guarded
Apple-native paths.

Version 0.1.0 is a developer preview. It is deliberately conservative: every
native route has device, dtype, shape, and runtime guards, and unsupported work
falls back to normal Forge behavior.

## What is accelerated today

| Area | v0.1 route | Current evidence | Status |
| --- | --- | --- | --- |
| Normal Anima generation | Metal RoPE + MPSGraph self/cross attention | qualified visual-parity ladder; large cumulative gain over vanilla Forge on the development Mac | enabled in `visual-fast` |
| Optional normal-generation runtime | pinned PyTorch nightly + exact Torch-2.12 compatibility kernels | cooled 10/15/30-step reductions of **16.49% / 14.96% / 13.26%** versus the already accelerated route | opt-in preview launcher |
| Hi-Res second denoiser | shape-gated 9216-token MPSGraph attention | representative sampler-loop improvement of about **218s → 144s** | component qualified; new whole-Hi-Res gate pending |
| SwinIR/ESRGAN resize | Forge's GPU tile compositor + measured Apple tile setting | controlled **60.687s → 43.111s** compositor comparison; later tuned SwinIR stage **34.502s** | enabled in `visual-fast` |
| Tiled/Ultimate upscale | groundwork only | no complete promoted path yet | later phase |

The 13–16% nightly result does **not** mean SwinIR itself is 16% faster, and it
does not yet mean the complete Hi-Res request is 16% faster. On the accepted
1024→1536 profile, the second denoiser took 138.276s and SwinIR took 34.502s.
Applying the normal-generation runtime gain only to that denoiser projects an
18–22s saving, roughly 10–12% of the measured stage total. That projection must
still pass a cooled, same-seed whole-Hi-Res A/B test.

See [BENCHMARKS.md](BENCHMARKS.md) for the exact scope and methodology.

## Requirements

- Apple Silicon Mac (`arm64`)
- macOS with Xcode Command Line Tools
- Forge Neo / Forge Classic `neo` branch
- tested Forge commit: `e91c293170e4451fc27490827f9aa62ccc77e6dc`
- Forge's current Python 3.13 environment

Other hosts are left untouched. Forge updates that change the patched stock
Anima method signatures are rejected safely and reported by the status API.

## Install from Forge Neo

Once the repository is public:

1. Open **Extensions → Install from URL**.
2. Paste `https://github.com/Gohro/forge-neo-apple-accelerator`.
3. Install, open the **Installed** tab, then choose **Apply and restart UI**.

Forge clones URL-installed extensions into `extensions/`, runs `install.py`,
and asks for a full restart. The add-on's preload hook runs early on the next
process and installs either Forge provider API v1 or the guarded stock adapter.

For a ZIP installation, extract the `forge_apple_accelerator` folder into
Forge's `extensions/` directory, run its `install.py` with Forge's Python, and
fully restart Forge:

```bash
venv/bin/python extensions/forge_apple_accelerator/install.py
```

The native bridge is compiled locally because it is fingerprinted to the exact
Torch, Python, macOS, extension source, and Forge integration in use. Generated
binaries are intentionally not shipped in the repository or ZIP.

## Modes and settings

Open **Settings → Apple Accelerator** after installation.

- `off`: normal Forge/PyTorch paths.
- `strict`: exact Metal RoPE only.
- `visual-fast` (default): Metal RoPE, MPSGraph attention, GPU upscaler
  compositing, and the configured Apple tile size.
- `inherit`: preserve environment settings from an external launcher.

The default upscaler tile is 768, which selects the qualified large-tile or
full-model behavior on the development 48 GB Mac. Use 512 on lower-memory Macs.
Mode and tile changes require a full backend restart.

## Optional Nightly Exact Preview

The extra 13–16% normal-generation gain is process-wide and cannot be activated
by hot-reloading an extension. It uses an immutable overlay owned by this
extension; Forge's main `venv` is never modified.

Install the pinned, hash-verified overlay once:

```bash
venv/bin/python extensions/forge_apple_accelerator/install_runtime.py
```

Then start Forge with:

```bash
extensions/forge_apple_accelerator/launch_apple_accelerated.sh
```

The launcher shadows only Torch and TorchVision, builds a separate bridge for
that exact nightly, enables the compatibility kernels, and invokes Forge with
`--skip-prepare-environment --skip-install`. Standard Forge launches continue
to use the normal venv and stable bridge.

The overlay is pinned to:

- `torch 2.15.0.dev20260901`
- `torchvision 0.30.0.dev20260901`
- CPython 3.13 / macOS arm64

It is not downloaded during normal extension installation.

## Status and self-test

Runtime status:

```text
GET /internal/forge-apple-accelerator/status
```

Synthetic Metal/MPSGraph gate:

```bash
PYTHONPATH=extensions/forge_apple_accelerator \
  venv/bin/python -m forge_apple_accelerator.self_test
```

The 2026-09-03 release-source gate passed exact Metal RoPE, MPSGraph attention
parity, unsupported-shape fallback, and 100-call memory stability.

## Disable or remove

Disable the extension in Forge's **Installed** tab and restart. The optional
runtime is used only by `launch_apple_accelerated.sh`; a standard Forge launch
never imports it. Deleting `runtime_overlay/` removes that optional runtime.

## Known limits

- Whole-request validation of v0.1 inside an entirely pristine stock checkout
  is still required before labeling this a stable release.
- The 9216-token attention and SwinIR wins are component-qualified; a new cooled
  complete Hi-Res A/B/visual gate is the next milestone.
- The exact Ultimate SD Upscale extension is not installed in the development
  checkout and is not claimed as accelerated.
- The nightly overlay is intentionally pinned, not automatically updated.
- Visual parity always means same checkpoint, prompt, negative prompt, seed,
  sampler, scheduler, dimensions, CFG, and step count, followed by image review;
  numerical tensor gates alone are not accepted as proof.

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
