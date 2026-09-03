# Changelog

## Unreleased

- Requalified the stable add-on against literal vanilla Forge in fresh
  10/15/30-step ABBA runs and replaced historical single-sample headline
  numbers with the new two-sample medians.
- Separated default stable gains from the optional exact-nightly incremental
  results, documented the enable/disable restart lifecycle, and stopped short
  of a combined claim until a same-campaign three-route matrix exists.
- Added exact-nightly install/activation and loaded-Torch status to the Settings
  panel and status API.
- Redesigned the project overview around a concise support matrix, measured
  gains, and three-step installation flow.
- Clarified that v0.1 normal text-to-image acceleration is Anima-only.
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
