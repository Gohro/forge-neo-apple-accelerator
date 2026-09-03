# Benchmark evidence

All promoted results below were measured on the same 48 GB Apple Silicon
development Mac. Low Power Mode was off. Promoted A/B comparisons used cooldowns
and checked macOS thermal state at launch and request boundaries.

## Default install: stable add-on versus vanilla

The 2026-09-03 release matrix compared literal `off` with stable Torch 2.12.1
`visual-fast`. It used fresh Forge processes, a one-step model-load warm-up,
two samples per route per step count, `off → visual-fast → visual-fast → off`
order, at least 45 seconds of cooldown, and a nominal thermal-state requirement
before every process. The first 30-step vanilla request ended in `fair`; the
sequence was stopped until the Mac returned to `nominal` before continuing.

| Steps | Vanilla samples / median | Stable add-on samples / median | Median reduction | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 29.397, 37.228s / **33.313s** | 17.674, 22.237s / **19.956s** | **40.10%** | 1.67× |
| 15 | 52.697, 42.734s / **47.715s** | 25.394, 25.275s / **25.334s** | **46.91%** | 1.88× |
| 30 | 88.302, 83.096s / **85.699s** | 48.757, 48.744s / **48.751s** | **43.11%** | 1.76× |

The checkpoint was `novaCartoonAM_v10` (`1eacc855eb`) with the Qwen image VAE
and Qwen 3 0.6B text encoder. Every run used 1024×1024, seed 12345, GPU RNG,
Euler a, Normal scheduler, CFG 4, and distilled CFG/shift 3. Status telemetry
showed provider-disabled fallback for all vanilla operations and live Metal
RoPE plus MPSGraph self/cross-attention hits for every `visual-fast` run.

Prompt: `masterpiece, best quality, score_7, safe, 1girl, solo, brown hair,
green eyes, school uniform, soft smile, looking at viewer, clean lineart,
simple background`. Negative prompt: `worst quality, low quality, score_1,
score_2, score_3, artist name`.

Both vanilla repeats and both `visual-fast` repeats were pixel-exact within
each step count. Vanilla versus `visual-fast` was not pixel-exact: SSIM was
0.8149 at 10 steps, 0.7462 at 15, and 0.8634 at 30. Full-resolution review
found both routes visually strong and closely matched in subject, face,
linework, palette, and prompt adherence, with differences in clothing and
small composition details. This qualifies `visual-fast`, not strict seed
identity.

The public [machine-readable evidence](evidence/anima-stable-release-matrix-20260903.json)
contains the retained result summary. The full local raw report is
`models/apple_mps/benchmarks/phase3-addon-release-abba-20260903/release-matrix-report.json`.

The earlier July 38.47→19.74s, 55.95→28.39s, and 120.84→68.83s ladder remains
historical development evidence. Each row had one sample and the 30-step
baseline came from an earlier run; those values are no longer used as the
release headline.

## Optional exact-nightly versus stable add-on

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

These percentages cannot be added directly to the stable-versus-vanilla table.
The combined reduction would be `1 - (stable/vanilla) × (nightly/stable)`, and
it must use measurements from the same counterbalanced three-route campaign.
No current combined vanilla-to-nightly number is claimed.

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

## Model coverage: Nova Comic XL V10 (SDXL)

The 2026-09-03 coverage gate used Nova Comic XL V10 at 1024×1024, 10 Euler a
steps, Normal scheduler, CFG 5, seed 12345, fresh Forge processes, an
`off → visual-fast → visual-fast → off` order, 45-second cooldowns, and nominal
macOS thermal state at every request boundary.

The decisive route result was consistent across all sessions:

- `provider.loaded_runtime_modules` remained empty;
- `rope_pair` and `attention_core` recorded no route hits;
- the whole-denoiser provider reported `provider_disabled` fallback;
- the checkpoint was identified as `novaComicXL_v10` with hash `a7c35838fe`.

Therefore the current add-on performs no accelerated base-generation work for
SDXL. No SDXL speedup is claimed.

Two exploratory timing matrices were rejected. In the GPU-RNG matrix,
accelerator-off samples were `16.485s` and `18.615s`, while visual-fast samples
were `15.308s` and `15.239s`; the visual-fast images were visibly invalid and
the vanilla repeats were not deterministic. A CPU-RNG repeat produced off
samples of `18.710s` and `15.289s` and visual-fast samples of `18.605s` and
`18.878s`; three images were byte-identical but visibly invalid, while the
fourth differed. All reported thermal states remained nominal. These timings
are preserved as failure evidence, not performance evidence.

Public summary record:

- [`evidence/nova-comic-xl-v10-20260903.json`](evidence/nova-comic-xl-v10-20260903.json)

Full raw artifacts on the development host:

- `models/apple_mps/benchmarks/phase3-model-coverage-nova-comic-xl-v10-clean-20260903/`
- `models/apple_mps/benchmarks/phase3-model-coverage-nova-comic-xl-v10-cpu-rng-20260903/`

Product decision: v0.1 normal-generation acceleration is labeled **Anima
only**. The SwinIR GPU compositor remains model-agnostic when an SDXL Hi-Res or
upscale workflow actually invokes Forge's SwinIR/ESRGAN path.
