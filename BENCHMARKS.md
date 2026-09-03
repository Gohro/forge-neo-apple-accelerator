# Benchmark evidence

All promoted results below were measured on the same 48 GB Apple Silicon
development Mac. Low Power Mode was off. Promoted A/B comparisons used cooldowns
and checked macOS thermal state at launch and request boundaries.

## Normal generation

The optional exact-nightly runtime was compared with the already accelerated
Forge route, not with unmodified vanilla Forge:

| Steps | Accelerated Forge | Nightly exact | Reduction | PSNR | SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 25.07071s median | 20.93629s median | 16.491% | 55.7039 dB | 0.998532 |
| 15 | 35.44265s | 30.13923s | 14.963% | 55.4260 dB | 0.998403 |
| 30 | 67.94123s | 58.93344s | 13.258% | 54.1375 dB | 0.998117 |

All pairs used the same checkpoint, prompt, negative prompt, seed 12345, GPU
RNG semantics, Euler a sampler, Normal scheduler, dimensions, and CFG. The
denoiser inputs and outputs were byte-identical at all ten captured sampler
steps. The small final pixel delta was isolated to the newer VAE decode. The
user's full-resolution review judged all pairs visually indistinguishable.

The 10-step result is a cooled `Forge → nightly → nightly → Forge` ABBA median
with 60-second cooldowns. The 15- and 30-step rows are single cooled visual
pairs and are not multi-pair timing qualifications.

## Hi-Res Fix and SwinIR

Accepted 1024→1536 stage allocation:

| Stage | Time |
| --- | ---: |
| Second-pass sampler | 138.27618s |
| SwinIR model | 34.50200s |
| Final VAE decode | 5.50089s |
| Hi-Res VAE encode | 3.49412s |
| First-pass decode | 2.25203s |
| PIL/stack/upload around resize | about 0.132s |

The old CPU compositor bottleneck was reduced in a controlled comparison from
60.687s to 43.111s (29.0%) by using Forge's GPU tile compositor. A later large
tile/full-model run reached 35.786s, and the current measured SwinIR model stage
is 34.502s. That means the remaining resize cost is overwhelmingly model GPU
compute, not Python/PIL tiling.

At the 9216-token Hi-Res attention shape, a representative 25-call second-pass
sampler loop improved from roughly 218s to 144s with the guarded MPSGraph path.
This is a component result, not a complete-request percentage.

## Release-source self-test (2026-09-03)

With native Metal access, the source-only v0.1 package produced:

- exact Metal RoPE (`max_abs = 0` for q and k);
- MPSGraph 2304-token attention: 13.402ms candidate vs 63.381ms Torch SDPA
  reference (4.729× primitive speedup);
- finite output, max absolute delta 0.0009765625 and mean absolute delta
  0.000011772;
- zero allocated-memory growth across the repeated-call gate;
- correct fallback for an unsupported 64-token/non-contiguous input;
- guarded stock-Neo adapter smoke pass against pristine commit
  `e91c293170e4451fc27490827f9aa62ccc77e6dc`, including byte-exact CPU
  fallback before/after patching.

Synthetic primitive speedups are never substituted for full-generation timing.
