import os
import threading
import time
from typing import Any


_FALSE_VALUES = {"", "0", "false", "no", "off"}
_CACHE_LOCK = threading.Lock()
_SHADER_CACHE: dict[str, Any] = {}
_STATS = {
    "compile_count": 0,
    "compile_errors": 0,
    "calls": 0,
    "fallbacks": 0,
}


ADDCMUL_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

kernel void addcmul_f32(device const float* input,
                        device const float* tensor1,
                        device const float* tensor2,
                        device float* output,
                        uint idx [[thread_position_in_grid]]) {
    output[idx] = fma(tensor1[idx], tensor2[idx], input[idx]);
}

kernel void addcmul_f16(device const half* input,
                        device const half* tensor1,
                        device const half* tensor2,
                        device half* output,
                        uint idx [[thread_position_in_grid]]) {
    output[idx] = fma(tensor1[idx], tensor2[idx], input[idx]);
}
"""


ROPE_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

inline float bf16_to_float(ushort x) {
    return as_type<float>(((uint)x) << 16);
}

inline ushort float_to_bf16_rne(float x) {
    uint bits = as_type<uint>(x);
    uint lsb = (bits >> 16) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return ushort(rounded >> 16);
}

kernel void rope_pair_f32(device const float* xq,
                          device const float* xk,
                          device const float* freqs,
                          device float* q_out,
                          device float* k_out,
                          constant uint& sequence,
                          constant uint& heads,
                          constant uint& dim,
                          uint idx [[thread_position_in_grid]]) {
    const uint half_dim = dim / 2;
    const uint d = idx % dim;
    const uint s = (idx / (heads * dim)) % sequence;
    const uint pair_dim = d < half_dim ? d : d - half_dim;
    const uint paired_d = d < half_dim ? d + half_dim : d - half_dim;
    const uint freq_base = (s * half_dim + pair_dim) * 4;
    const float a = xq[idx];
    const float b = xq[idx + paired_d - d];
    const float c = xk[idx];
    const float e = xk[idx + paired_d - d];
    if (d < half_dim) {
        q_out[idx] = freqs[freq_base + 0] * a + freqs[freq_base + 1] * b;
        k_out[idx] = freqs[freq_base + 0] * c + freqs[freq_base + 1] * e;
    } else {
        q_out[idx] = freqs[freq_base + 2] * b + freqs[freq_base + 3] * a;
        k_out[idx] = freqs[freq_base + 2] * e + freqs[freq_base + 3] * c;
    }
}

kernel void rope_pair_f16(device const half* xq,
                          device const half* xk,
                          device const float* freqs,
                          device half* q_out,
                          device half* k_out,
                          constant uint& sequence,
                          constant uint& heads,
                          constant uint& dim,
                          uint idx [[thread_position_in_grid]]) {
    const uint half_dim = dim / 2;
    const uint d = idx % dim;
    const uint s = (idx / (heads * dim)) % sequence;
    const uint pair_dim = d < half_dim ? d : d - half_dim;
    const uint paired_d = d < half_dim ? d + half_dim : d - half_dim;
    const uint freq_base = (s * half_dim + pair_dim) * 4;
    const float a = float(xq[idx]);
    const float b = float(xq[idx + paired_d - d]);
    const float c = float(xk[idx]);
    const float e = float(xk[idx + paired_d - d]);
    if (d < half_dim) {
        q_out[idx] = half(freqs[freq_base + 0] * a + freqs[freq_base + 1] * b);
        k_out[idx] = half(freqs[freq_base + 0] * c + freqs[freq_base + 1] * e);
    } else {
        q_out[idx] = half(freqs[freq_base + 2] * b + freqs[freq_base + 3] * a);
        k_out[idx] = half(freqs[freq_base + 2] * e + freqs[freq_base + 3] * c);
    }
}

kernel void rope_pair_bf16(device const ushort* xq,
                           device const ushort* xk,
                           device const float* freqs,
                           device ushort* q_out,
                           device ushort* k_out,
                           constant uint& sequence,
                           constant uint& heads,
                           constant uint& dim,
                           uint idx [[thread_position_in_grid]]) {
    const uint half_dim = dim / 2;
    const uint d = idx % dim;
    const uint s = (idx / (heads * dim)) % sequence;
    const uint pair_dim = d < half_dim ? d : d - half_dim;
    const int paired_offset = d < half_dim ? int(half_dim) : -int(half_dim);
    const uint paired_idx = uint(int(idx) + paired_offset);
    const uint freq_base = (s * half_dim + pair_dim) * 4;

    const float f0 = freqs[freq_base + 0];
    const float f1 = freqs[freq_base + 1];
    const float f2 = freqs[freq_base + 2];
    const float f3 = freqs[freq_base + 3];
    const float cur_q = bf16_to_float(xq[idx]);
    const float pair_q = bf16_to_float(xq[paired_idx]);
    const float cur_k = bf16_to_float(xk[idx]);
    const float pair_k = bf16_to_float(xk[paired_idx]);

    float q;
    float k;
    if (d < half_dim) {
        float q0 = f0 * cur_q;
        float q1 = f1 * pair_q;
        float k0 = f0 * cur_k;
        float k1 = f1 * pair_k;
        q = q0 + q1;
        k = k0 + k1;
    } else {
        float q0 = f2 * pair_q;
        float q1 = f3 * cur_q;
        float k0 = f2 * pair_k;
        float k1 = f3 * cur_k;
        q = q0 + q1;
        k = k0 + k1;
    }

    q_out[idx] = float_to_bf16_rne(q);
    k_out[idx] = float_to_bf16_rne(k);
}
"""


# Reproduces the PyTorch 2.12.1 MPS ``layer_norm_single_row_bfloat`` forward
# algorithm for Anima's affine-free, 2048-wide LayerNorm.  Newer PyTorch uses a
# two-pass centered variance, which is more robust in general but changes BF16
# diffusion trajectories.  This compatibility kernel intentionally preserves
# the old E[x^2] - E[x]^2 arithmetic and is only exposed behind an explicit
# experimental route.
LEGACY_LAYER_NORM_SHADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float bf16_to_float(ushort x) {
    return as_type<float>(((uint)x) << 16);
}

inline ushort float_to_bf16_rne(float x) {
    uint bits = as_type<uint>(x);
    uint lsb = (bits >> 16) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return ushort(rounded >> 16);
}

kernel void legacy_layer_norm_bf16_2048(
        device const ushort* input [[buffer(0)]],
        device ushort* output [[buffer(1)]],
        constant uint& axis_size [[buffer(2)]],
        constant float& epsilon [[buffer(3)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint simd_lane_id [[thread_index_in_simdgroup]],
        uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_READS = 4;
    const uint row_offset = tg_id * axis_size;
    const uint base_lane = tid * N_READS;
    device const ushort* x = input + row_offset + base_lane;

    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    if (base_lane + N_READS <= axis_size) {
        float4 v4 = float4(
            bf16_to_float(x[0]),
            bf16_to_float(x[1]),
            bf16_to_float(x[2]),
            bf16_to_float(x[3]));
        partial_sum = v4.x + v4.y + v4.z + v4.w;
        partial_sum_sq = dot(v4, v4);
    }

    threadgroup float local_sums[32];
    threadgroup float local_sums_sq[32];
    threadgroup float tg_mean[1];
    threadgroup float tg_inv_std[1];
    if (simdgroup_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
        local_sums_sq[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float group_partial_sum = simd_sum(partial_sum);
    const float group_partial_sum_sq = simd_sum(partial_sum_sq);
    if (simd_lane_id == 0) {
        local_sums[simdgroup_id] = group_partial_sum;
        local_sums_sq[simdgroup_id] = group_partial_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id == 0) {
        const float sum = simd_sum(local_sums[simd_lane_id]);
        const float sum_sq = simd_sum(local_sums_sq[simd_lane_id]);
        if (simd_lane_id == 0) {
            const float mean = sum / float(axis_size);
            float variance = sum_sq / float(axis_size) - mean * mean;
            variance = variance < 1.0e-6f ? 0.0f : variance;
            tg_mean[0] = mean;
            tg_inv_std[0] = precise::rsqrt(variance + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = tg_mean[0];
    const float inv_std = tg_inv_std[0];
    if (base_lane + N_READS <= axis_size) {
        #pragma unroll
        for (uint i = 0; i < N_READS; ++i) {
            const float value = bf16_to_float(x[i]);
            output[row_offset + base_lane + i] =
                float_to_bf16_rne((value - mean) * inv_std);
        }
    }
}
"""


# Reproduces PyTorch 2.12's fused MPS BF16 RMSNorm. PyTorch 2.13+ changed
# multiplication order from ``w * bf16(x * inv_rms)`` to a float-accumulated
# ``bf16(x * inv_rms * w)``, which changes Anima conditioning and q/k bytes.
LEGACY_RMS_NORM_SHADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float bf16_to_float(ushort x) {
    return as_type<float>(((uint)x) << 16);
}

inline ushort float_to_bf16_rne(float x) {
    uint bits = as_type<uint>(x);
    uint lsb = (bits >> 16) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return ushort(rounded >> 16);
}

kernel void legacy_rms_norm_bf16(
        device const ushort* input [[buffer(0)]],
        device const ushort* weight [[buffer(1)]],
        device ushort* output [[buffer(2)]],
        constant uint& axis_size [[buffer(3)]],
        constant float& epsilon [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint simd_lane_id [[thread_index_in_simdgroup]],
        uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_READS = 4;
    const uint lane_base = tid * N_READS;
    const uint row_offset = row * axis_size;
    float acc = 0.0f;
    #pragma unroll
    for (uint i = 0; i < N_READS; ++i) {
        if (lane_base + i < axis_size) {
            const float value = bf16_to_float(input[row_offset + lane_base + i]);
            acc += value * value;
        }
    }

    acc = simd_sum(acc);
    threadgroup float local_sums[32];
    threadgroup float local_inv_rms[1];
    if (simdgroup_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simdgroup_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_rms[0] = precise::rsqrt(acc / float(axis_size) + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint i = 0; i < N_READS; ++i) {
        const uint column = lane_base + i;
        if (column < axis_size) {
            const float value = bf16_to_float(input[row_offset + column]);
            const ushort normalized_bf16 = float_to_bf16_rne(value * local_inv_rms[0]);
            const float product = bf16_to_float(weight[column]) * bf16_to_float(normalized_bf16);
            output[row_offset + column] = float_to_bf16_rne(product);
        }
    }
}
"""


ATTENTION_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

constant uint ATTN_BATCH = 2;
constant uint ATTN_SEQUENCE = 4096;
constant uint ATTN_HEADS = 16;
constant uint ATTN_DIM = 128;
constant uint ATTN_THREADS = 128;
constant float ATTN_SCALE = 0.08838834764831845f;
constant float ATTN_LOG2E = 1.4426950408889634f;

inline float bf16_to_float(ushort x) {
    return as_type<float>(((uint)x) << 16);
}

inline ushort float_to_bf16_rne(float x) {
    uint bits = as_type<uint>(x);
    uint lsb = (bits >> 16) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return ushort(rounded >> 16);
}

kernel void attention_core_bf16_2_4096_16_128(device const ushort* q,
                                              device const ushort* k,
                                              device const ushort* v,
                                              device ushort* out,
                                              uint gid [[thread_position_in_grid]],
                                              uint tid [[thread_position_in_threadgroup]]) {
    const uint row = gid / ATTN_THREADS;
    if (row >= ATTN_BATCH * ATTN_HEADS * ATTN_SEQUENCE) {
        return;
    }

    const uint query = row % ATTN_SEQUENCE;
    const uint head = (row / ATTN_SEQUENCE) % ATTN_HEADS;
    const uint batch = row / (ATTN_SEQUENCE * ATTN_HEADS);

    threadgroup float scores[ATTN_SEQUENCE];
    threadgroup float scratch[ATTN_THREADS];

    float local_max = -INFINITY;
    for (uint key = tid; key < ATTN_SEQUENCE; key += ATTN_THREADS) {
        float dot = 0.0f;
        const uint q_base = ((batch * ATTN_SEQUENCE + query) * ATTN_HEADS + head) * ATTN_DIM;
        const uint k_base = ((batch * ATTN_SEQUENCE + key) * ATTN_HEADS + head) * ATTN_DIM;
        for (uint d = 0; d < ATTN_DIM; ++d) {
            dot += bf16_to_float(q[q_base + d]) * bf16_to_float(k[k_base + d]);
        }
        const float score = dot * ATTN_SCALE;
        scores[key] = score;
        local_max = max(local_max, score);
    }

    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ATTN_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];

    float local_sum = 0.0f;
    for (uint key = tid; key < ATTN_SEQUENCE; key += ATTN_THREADS) {
        const float weight = exp2((scores[key] - row_max) * ATTN_LOG2E);
        scores[key] = weight;
        local_sum += weight;
    }

    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ATTN_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = 1.0f / scratch[0];

    if (tid < ATTN_DIM) {
        float acc = 0.0f;
        for (uint key = 0; key < ATTN_SEQUENCE; ++key) {
            const float weight = scores[key] * inv_sum;
            const uint v_index = ((batch * ATTN_SEQUENCE + key) * ATTN_HEADS + head) * ATTN_DIM + tid;
            acc += weight * bf16_to_float(v[v_index]);
        }
        const uint out_index = (batch * ATTN_SEQUENCE + query) * (ATTN_HEADS * ATTN_DIM) + head * ATTN_DIM + tid;
        out[out_index] = float_to_bf16_rne(acc);
    }
}
"""


def mode() -> str:
    return os.getenv("FORGE_APPLE_METAL_KERNELS", "off").strip().lower()


def enabled_for(kind: str) -> bool:
    current = mode()
    if current in _FALSE_VALUES:
        return False
    if current in {"1", "true", "yes", "on", "all"}:
        return True
    if current == "probe":
        return kind in {"probe", "pointwise", "addcmul"}
    if current == "pointwise":
        return kind in {"pointwise", "addcmul"}
    return current == kind


def _enabled_value(name: str) -> bool:
    return os.getenv(name, "").strip().lower() not in _FALSE_VALUES


def stats() -> dict[str, Any]:
    return dict(_STATS)


def _torch():
    import torch

    return torch


def _compile(name: str, source: str):
    with _CACHE_LOCK:
        if name in _SHADER_CACHE:
            return _SHADER_CACHE[name]

    torch = _torch()
    if not hasattr(torch, "mps") or not hasattr(torch.mps, "compile_shader"):
        _STATS["fallbacks"] += 1
        return None
    if not torch.backends.mps.is_available():
        _STATS["fallbacks"] += 1
        return None

    started = time.perf_counter()
    try:
        library = torch.mps.compile_shader(source)
    except Exception:
        _STATS["compile_errors"] += 1
        raise

    with _CACHE_LOCK:
        _SHADER_CACHE[name] = library
        _SHADER_CACHE[f"{name}:compile_s"] = time.perf_counter() - started
        _STATS["compile_count"] += 1
    return library


def _is_supported_tensor(tensor, dtype_names: set[str]) -> bool:
    return (
        hasattr(tensor, "is_mps")
        and tensor.is_mps
        and str(tensor.dtype) in dtype_names
        and tensor.is_contiguous()
    )


def _group_size(numel: int) -> tuple[int, int, int]:
    return (min(256, max(1, numel)), 1, 1)


def _mps_memory_snapshot() -> dict[str, int | None]:
    torch = _torch()
    if not hasattr(torch, "mps"):
        return {}
    snapshot = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        fn = getattr(torch.mps, name, None)
        if fn is None:
            snapshot[name] = None
            continue
        try:
            snapshot[name] = int(fn())
        except Exception:
            snapshot[name] = None
    return snapshot


def addcmul(input, tensor1, tensor2):
    if not enabled_for("addcmul"):
        return None
    dtype_name = str(input.dtype)
    if dtype_name not in {"torch.float32", "torch.float16"}:
        _STATS["fallbacks"] += 1
        return None
    if not (_is_supported_tensor(input, {dtype_name}) and _is_supported_tensor(tensor1, {dtype_name}) and _is_supported_tensor(tensor2, {dtype_name})):
        _STATS["fallbacks"] += 1
        return None

    library = _compile("addcmul", ADDCMUL_SHADER)
    if library is None:
        return None
    output = _torch().empty_like(input)
    kernel = library.addcmul_f32 if dtype_name == "torch.float32" else library.addcmul_f16
    _STATS["calls"] += 1
    kernel(input, tensor1, tensor2, output, threads=(input.numel(), 1, 1), group_size=_group_size(input.numel()))
    return output


def rope_pair(xq, xk, freqs):
    if not enabled_for("rope"):
        return None
    dtype_name = str(xq.dtype)
    if dtype_name not in {"torch.float32", "torch.float16", "torch.bfloat16"}:
        _STATS["fallbacks"] += 1
        return None
    if xq.ndim != 4 or xk.shape != xq.shape or xq.shape[-1] % 2 != 0:
        _STATS["fallbacks"] += 1
        return None
    if not (_is_supported_tensor(xq, {dtype_name}) and _is_supported_tensor(xk, {dtype_name})):
        _STATS["fallbacks"] += 1
        return None
    if not hasattr(freqs, "is_mps") or not freqs.is_mps:
        _STATS["fallbacks"] += 1
        return None

    torch = _torch()
    batch, sequence, heads, dim = xq.shape
    half_dim = dim // 2
    if freqs.numel() != sequence * half_dim * 4:
        _STATS["fallbacks"] += 1
        return None

    freqs_flat = freqs.reshape(sequence, half_dim, 2, 2).contiguous().to(dtype=torch.float32)
    library = _compile("rope", ROPE_SHADER)
    if library is None:
        return None
    q_out = torch.empty_like(xq)
    k_out = torch.empty_like(xk)
    if dtype_name == "torch.float32":
        kernel = library.rope_pair_f32
    elif dtype_name == "torch.float16":
        kernel = library.rope_pair_f16
    else:
        kernel = library.rope_pair_bf16
    _STATS["calls"] += 1
    kernel(
        xq,
        xk,
        freqs_flat,
        q_out,
        k_out,
        sequence,
        heads,
        dim,
        threads=(xq.numel(), 1, 1),
        group_size=_group_size(xq.numel()),
    )
    return q_out, k_out


def legacy_layer_norm(input_tensor, eps: float = 1.0e-6, *, allow_experimental: bool = False):
    if not allow_experimental and os.getenv("FORGE_APPLE_LAYER_NORM_BACKEND", "off").strip().lower() != "legacy-metal":
        return None
    dtype_name = str(input_tensor.dtype)
    if dtype_name != "torch.bfloat16" or input_tensor.ndim < 2 or input_tensor.shape[-1] != 2048:
        _STATS["fallbacks"] += 1
        return None
    if not _is_supported_tensor(input_tensor, {dtype_name}):
        _STATS["fallbacks"] += 1
        return None

    torch = _torch()
    library = _compile("legacy_layer_norm_bf16_2048", LEGACY_LAYER_NORM_SHADER)
    if library is None:
        return None
    output = torch.empty_like(input_tensor)
    axis_size = 2048
    rows = input_tensor.numel() // axis_size
    threads_per_group = axis_size // 4
    _STATS["calls"] += 1
    library.legacy_layer_norm_bf16_2048(
        input_tensor,
        output,
        axis_size,
        float(eps),
        threads=(rows * threads_per_group, 1, 1),
        group_size=(threads_per_group, 1, 1),
    )
    return output


def legacy_rms_norm(input_tensor, weight, eps: float = 1.0e-6, *, allow_experimental: bool = False):
    if not allow_experimental and os.getenv("FORGE_APPLE_RMS_NORM_BACKEND", "off").strip().lower() != "legacy-metal":
        return None
    dtype_name = str(input_tensor.dtype)
    if dtype_name != "torch.bfloat16" or input_tensor.ndim < 1:
        _STATS["fallbacks"] += 1
        return None
    axis_size = int(input_tensor.shape[-1])
    if axis_size not in {128, 2048} or tuple(weight.shape) != (axis_size,) or str(weight.dtype) != dtype_name:
        _STATS["fallbacks"] += 1
        return None
    if not (_is_supported_tensor(input_tensor, {dtype_name}) and _is_supported_tensor(weight, {dtype_name})):
        _STATS["fallbacks"] += 1
        return None

    torch = _torch()
    library = _compile("legacy_rms_norm_bf16", LEGACY_RMS_NORM_SHADER)
    if library is None:
        return None
    output = torch.empty_like(input_tensor)
    rows = input_tensor.numel() // axis_size
    threads_per_group = 32 * ((axis_size + 127) // 128)
    _STATS["calls"] += 1
    library.legacy_rms_norm_bf16(
        input_tensor,
        weight,
        output,
        axis_size,
        float(eps),
        threads=(rows * threads_per_group, 1, 1),
        group_size=(threads_per_group, 1, 1),
    )
    return output


def attention_core(q, k, v, *, is_self_attention: bool = True, allow_experimental: bool = False):
    if not enabled_for("attention"):
        return None
    if not allow_experimental and not _enabled_value("FORGE_APPLE_METAL_ATTENTION_RUNTIME"):
        _STATS["fallbacks"] += 1
        return None
    if not is_self_attention:
        _STATS["fallbacks"] += 1
        return None
    dtype_name = str(q.dtype)
    if dtype_name != "torch.bfloat16":
        _STATS["fallbacks"] += 1
        return None
    if q.shape != (2, 4096, 16, 128) or k.shape != q.shape or v.shape != q.shape:
        _STATS["fallbacks"] += 1
        return None
    if not (_is_supported_tensor(q, {dtype_name}) and _is_supported_tensor(k, {dtype_name}) and _is_supported_tensor(v, {dtype_name})):
        _STATS["fallbacks"] += 1
        return None

    torch = _torch()
    library = _compile("attention_core_bf16_2_4096_16_128", ATTENTION_SHADER)
    if library is None:
        return None
    output = torch.empty((2, 4096, 16 * 128), dtype=q.dtype, device=q.device)
    _STATS["calls"] += 1
    threads_per_group = 128
    rows = 2 * 4096 * 16
    library.attention_core_bf16_2_4096_16_128(
        q,
        k,
        v,
        output,
        threads=(rows * threads_per_group, 1, 1),
        group_size=(threads_per_group, 1, 1),
    )
    return output
