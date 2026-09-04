# Changelog

## Unreleased

## 0.2.0 - 2026-09-04

- Added a guarded BF16 compatibility route for the qualified
  `SwinIR_4x.pth` checkpoint, retaining FP32 softmax and falling back for other
  SwinIR variants.
- Added automatic stable Torch 2.14 compiled-SwinIR with persistent cache,
  eager fallback, and live status telemetry.
- Added an early startup bootstrap that selects the verified extension runtime
  before Forge imports Torch; disabling the add-on restores the original Forge
  runtime after restart.
- Pinned official stable Torch/TorchVision wheels by URL, size, and SHA-256;
  the installer keeps them in an extension-owned overlay.
- Qualified the packaged Hi-Res route in a cooled counterbalanced matrix:
  complete request median `134.49s → 86.88s` and SwinIR median
  `42.57s → 12.40s`.
- Requalified normal Anima generation on the packaged v0.2 route:
  `22.74s → 17.30s` median (`23.9%`) with nominal thermal boundaries and
  pixel-exact repeats within each route.
- Validated stable Torch 2.14 native VAE conv3d in a complete 10+10 Hi-Res
  workflow without modifying Forge's installed venv.
- Added fresh-process Hi-Res SwinIR/runtime benchmark harnesses and corrected
  cross-runtime parity tests to use Forge's CPU-noise semantics.
- Removed the unpaired historical `218s → 144s` Hi-Res sampler statement from
  the public performance claims.

- Requalified the stable add-on against literal vanilla Forge in fresh
  10/15/30-step ABBA runs and replaced historical single-sample headline
  numbers with the new two-sample medians.
- Replaced the former optional-nightly UX with one automatic supported route
  controlled by Forge's normal enable/disable extension lifecycle.
- Redesigned the project overview around a concise support matrix, measured
  gains, and three-step installation flow.
- Clarified that v0.2 normal text-to-image acceleration is Anima-only.
- Added the Nova Comic XL V10 SDXL coverage result: no native base-generation
  route activated, so no SDXL speedup is claimed.
- Added `MODEL_SUPPORT.md` and documented the model-agnostic boundary of the
  SwinIR GPU compositor.

## 0.1.0 - 2026-09-03

- Added guarded stock Forge Neo Anima adapter; provider API v1 remains preferred
  automatically when available.
- Added Metal RoPE and MPSGraph attention runtime with exact runtime manifests.
- Added GPU upscaler compositor and configurable Apple tile settings.
- Added optional, extension-owned PyTorch nightly overlay and exact-compatibility
  launcher without modifying Forge's venv.
- Added status API, settings panel, source self-test, pristine-stock smoke test,
  and source-only packaging rules.
- Documented the boundary between normal-generation, Hi-Res attention, SwinIR,
  and tiled-upscale claims.
