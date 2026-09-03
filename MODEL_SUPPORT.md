# Model support

Version 0.1 is intentionally narrow. A model is listed as accelerated only
after the route activates on that architecture and passes controlled timing
and visual review.

## Support matrix

| Model family / workflow | Base generation | Hi-Res denoiser | SwinIR upscale |
| --- | --- | --- | --- |
| Anima | **Accelerated** | Component-qualified | Supported |
| SDXL, including Nova Comic XL | Not accelerated | Not accelerated by the Anima route | Supported |
| Illustrious-family SDXL | Not accelerated | Not accelerated by the Anima route | Supported |
| Other Forge model families | Not claimed | Not claimed | GPU compositor may apply when Forge uses SwinIR/ESRGAN |

“Supported” in the SwinIR column means the add-on enables Forge's GPU tile
compositor and measured Apple tile setting. It does not mean the checkpoint's
denoiser is accelerated.

## Why Anima is faster

The current normal-generation routes target operations and shapes in Forge's
Anima implementation:

- exact BF16 Metal rotary position embedding;
- guarded MPSGraph self-attention;
- guarded MPSGraph cross-attention;
- an optional pinned runtime with Anima exact-compatibility kernels.

Every route checks device, dtype, shape, and runtime compatibility before it
runs. Unsupported work falls back instead of forcing an incompatible kernel.

## SDXL result

Nova Comic XL V10 was tested on 2026-09-03 at 1024×1024 and 10 steps with
fresh Forge processes, counterbalanced order, 45-second cooldowns, and nominal
thermal state. The accelerator status endpoint reported no loaded native
runtime modules and no SDXL kernel hits. Therefore v0.1 makes **no SDXL
base-generation speed claim**.

Some exploratory Forge/MPS samples also produced inconsistent VAE output, so
their elapsed times were rejected rather than presented as a speed result.
The preserved settings, samples, route telemetry, and image comparisons are
documented in [BENCHMARKS.md](BENCHMARKS.md).

## Roadmap

1. Keep the qualified Anima normal-generation path stable.
2. Complete whole-request Anima Hi-Res qualification.
3. Profile SDXL separately and build SDXL-native routes only where a measured
   bottleneck and parity-safe kernel justify them.
4. Qualify tiled and Ultimate-style upscale workflows end to end.
