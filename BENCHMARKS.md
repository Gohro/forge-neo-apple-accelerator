# Benchmark evidence

All promoted results below were measured on the same 48 GB Apple Silicon
development Mac. Low Power Mode was off. Promoted A/B comparisons used cooldowns
and checked macOS thermal state at launch and request boundaries.

## Normal Anima generation: v0.2 packaged route

The current normal-generation headline uses the same public lifecycle as the
Hi-Res release matrix: ordinary Forge launch, no manual route environment
overrides, and extension enable/disable state as the only switch. The
counterbalanced order was `accelerated → vanilla → vanilla → accelerated`,
with fresh processes, at least 55 seconds of cooldown, and nominal thermal
state at every request boundary.

At 1024×1024, 10 Euler a steps, CPU noise, and seed 12345, vanilla samples were
22.728s and 22.756s (22.742s median). Automatic v0.2 samples were 16.284s and
18.325s (17.305s median): **23.91% less time, 1.31× throughput, and 5.437s
saved**. Status proved Torch 2.14, a valid native bridge, Metal RoPE, and
MPSGraph attention on both candidate runs. The baseline loaded original Forge
Torch 2.12 and exposed no accelerator endpoint.

Both same-route pairs were pixel-exact. Cross-route SSIM was 0.930838, with the
same subject, composition, detail level, and visual quality under full review.

- [`evidence/anima-v02-release-matrix-20260904.json`](evidence/anima-v02-release-matrix-20260904.json)

## Historical v0.1 Torch 2.12 matrix

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

## Hi-Res Fix release matrix

The v0.2 public route was measured exactly as users activate it: the baseline
listed the extension in Forge's `disabled_extensions`, while the candidate used
an ordinary Forge launch with no manual accelerator environment overrides. The
candidate's early bootstrap selected the extension-owned stable Torch 2.14
runtime, and its preload selected BF16/compiled SwinIR and GPU compositing.

The counterbalanced order was
`accelerated → vanilla → vanilla → accelerated`, with fresh Forge processes,
at least 120 seconds of cooldown, and a nominal start requirement. Every run
used the same Anima checkpoint, prompt, CPU noise, seed 12345, 1024→1536,
10+10 Euler a steps, denoising strength 0.15, and `SwinIR_4x.pth`.

| Measurement | Vanilla samples / median | Automatic add-on samples / median | Median reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Complete request | 141.561, 127.418s / **134.489s** | 91.783, 81.984s / **86.883s** | **35.40%** | **1.55×** |
| SwinIR model | 45.316, 39.832s / **42.574s** | 13.234, 11.575s / **12.404s** | **70.86%** | **3.43×** |

Both accelerated sessions proved stable Torch 2.14, a hash-valid bridge,
guarded BF16, and compiled SwinIR with zero eager fallbacks. They remained
thermally nominal. Both longer vanilla sessions began nominal and ended fair;
this asymmetry is retained in the public evidence because it reflects the
additional sustained work rather than being silently normalized away.

The vanilla repeats were pixel-exact to each other, and the accelerated repeats
were pixel-exact to each other. Cross-route SSIM was 0.918194. Full-resolution
review found the same subject, composition, detail level, and overall quality,
with minor line and highlight placement drift. The route qualifies for visual
parity, not strict cross-runtime seed identity.

The clean-install first request separately measured 112.467s total and 35.264s
in SwinIR while the persistent graph cache was built. The next fresh process
used the cache and produced the exact same image bytes while reducing those
values to 91.783s and 13.234s.

Public release record:

- [`evidence/hires-release-matrix-20260904.json`](evidence/hires-release-matrix-20260904.json)

### Exact 25+25 user-workload confirmation

The user's observed shorter pause before the second progress bar was replayed
from the exact saved PNG metadata: 1024→1536, 25+25 Euler a steps, denoising
strength 0.3, seed 84553563, GPU RNG, and `SwinIR_4x.pth`. The order was
`accelerated → vanilla → vanilla → accelerated`, with fresh processes, at
least 120 seconds of cooldown, and nominal starts.

| Measurement | Add-on disabled median | Add-on enabled median | Improvement |
| --- | ---: | ---: | ---: |
| Complete request | 241.19s | 195.17s | **19.1% less / 1.24×** |
| Hi-Res start → second sampler | 42.75s | 15.10s | **64.7% less / 2.83×** |
| SwinIR full-model compute | 38.28s | 12.82s | **66.5% less / 2.99×** |
| Complete resize loop | 38.90s | 13.80s | **64.5% less / 2.82×** |
| Hi-Res VAE encode | 3.76s | 1.22s | **67.6% less** |
| Second sampler | 143.08s | 133.31s | **6.8% less** |

Both repeats were pixel-exact within each route. The user's original UI image
and the instrumented accelerated replay were visually indistinguishable (SSIM
0.99509). Cross-runtime GPU-RNG images are not a valid parity comparison
because Torch 2.12 and 2.14 use different MPS RNG streams; the CPU-RNG release
matrix above remains the cross-runtime visual gate.

The profiler also proves why no terminal tile progress appeared. On the 48 GB
machine, tile 768 and a 1024 source produce four planned tiles; Forge's planner
selects one full-model MPS call when the count is four or fewer. Neither the
CPU tile loop nor the GPU tile loop executes. Lower-memory Macs retain the
conservative tiled path.

- [`evidence/hires-user-workload-25x25-20260904.json`](evidence/hires-user-workload-25x25-20260904.json)

## Hi-Res investigation details

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

On 2026-09-04, the exact `SwinIR_4x.pth` checkpoint was measured with the full
1024→4096 model workload. A guarded BF16 patch keeps window softmax in FP32 and
casts probabilities back before the BF16 value matmul. In Forge on Torch
2.12.1 it reduced `upscale.gpu.full_model` from 33.922s to 23.785s (29.9%). The
corresponding 10+10 Hi-Res request fell from 147.143s to 137.519s. Final SSIM
was 0.996785 and the labeled pair was visually indistinguishable.

Stable Torch 2.14 does not improve this checkpoint in FP32 (`33.513s` on 2.12
versus `34.448s` on 2.14 when rested), because its head dimension is 30 and is
outside the new MPP-attention fast-path set. Torch 2.14 does accelerate the VAE:
in a CPU-noise 10+10 pair, decode+encode+decode fell from 12.650s to 3.797s.

With persistent compiled SwinIR graphs on Torch 2.14, the full Forge model stage
fell from 27.680s eager to 13.358s on the first request after restart, and the
complete request fell from 107.534s to 93.528s (13.0%). Standalone steady state
was 9.628s. The first-ever compile took about 36s per graph variant, while a
fresh process using the persistent cache loaded those variants in 13.198s and
13.035s. Eager/compiled final SSIM was 0.997750 and the full-resolution pair was
visually indistinguishable.

The historical `218s → 144s` second-pass statement is not a paired A/B and is
therefore not used as a release claim.

Earlier component results and lifecycle caveats are recorded in
[`evidence/hires-swinir-torch214-20260904.json`](evidence/hires-swinir-torch214-20260904.json).

## Release-source self-test

With native Metal access, the source package produced:

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

Product decision: v0.2 normal-generation acceleration is labeled **Anima
only**. The SwinIR GPU compositor remains model-agnostic when an SDXL Hi-Res or
upscale workflow actually invokes Forge's SwinIR/ESRGAN path.
