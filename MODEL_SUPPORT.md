# Model support

Version 0.2 is intentionally narrow. A model is listed as accelerated only
after the route activates on that architecture and passes controlled timing
and visual review.

## Support matrix

| Model family / workflow | Base generation | Hi-Res denoiser | SwinIR upscale |
| --- | --- | --- | --- |
| Anima | **Accelerated** | **Accelerated** | **BF16 + compiled `SwinIR_4x.pth`** |
| SDXL, including Nova Comic XL | Not accelerated | Not accelerated by the Anima route | GPU compositor; BF16 only for `SwinIR_4x.pth` |
| Illustrious-family SDXL | Not accelerated | Not accelerated by the Anima route | GPU compositor; BF16 only for `SwinIR_4x.pth` |
| Other Forge model families | Not claimed | Not claimed | GPU compositor may apply; BF16 is checkpoint-guarded |

The GPU compositor and Apple tile setting are architecture-independent. The
new BF16 and compiled graph routes are intentionally narrower: only the exact
standard `SwinIR_4x.pth` checkpoint is currently qualified. They do not imply
that the checkpoint's denoiser is accelerated.

## Why Anima is faster

The current normal-generation routes target operations and shapes in Forge's
Anima implementation:

- exact BF16 Metal rotary position embedding;
- guarded MPSGraph self-attention;
- guarded MPSGraph cross-attention;
- stable Torch 2.14 VAE execution from a verified extension-owned overlay.

Every route checks device, dtype, shape, and runtime compatibility before it
runs. Unsupported work falls back instead of forcing an incompatible kernel.

## SDXL result

Nova Comic XL V10 was tested on 2026-09-03 at 1024×1024 and 10 steps with
fresh Forge processes, counterbalanced order, 45-second cooldowns, and nominal
thermal state. The accelerator status endpoint reported no loaded native
runtime modules and no SDXL kernel hits. Therefore v0.2 makes **no SDXL
base-generation speed claim**.

Some exploratory Forge/MPS samples also produced inconsistent VAE output, so
their elapsed times were rejected rather than presented as a speed result.
The preserved settings, samples, route telemetry, and image comparisons are
documented in [BENCHMARKS.md](BENCHMARKS.md).

## Roadmap

1. Keep the qualified Anima normal-generation path stable.
2. Extend the released Anima Hi-Res matrix to more memory tiers and dimensions.
3. Profile SDXL separately and build SDXL-native routes only where a measured
   bottleneck and parity-safe kernel justify them.
4. Qualify tiled and Ultimate-style upscale workflows end to end.
