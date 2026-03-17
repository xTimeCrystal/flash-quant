from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cupy as cp
import torch


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


require(hasattr(torch, "float8_e4m3fn"), "Your torch build lacks torch.float8_e4m3fn.")
require(hasattr(torch, "float8_e8m0fnu"), "Your torch build lacks torch.float8_e8m0fnu.")
require(hasattr(torch, "float4_e2m1fn_x2"), "Your torch build lacks torch.float4_e2m1fn_x2.")

FP4_DTYPE = torch.float4_e2m1fn_x2
FP8_SCALE_DTYPE = torch.float8_e4m3fn


def _pick_row_threads(cols: int) -> int:
    if cols >= 32768:
        return 1024
    if cols >= 8192:
        return 512
    if cols >= 2048:
        return 256
    if cols >= 512:
        return 128
    return 64


# ============================================================
# INT8
# ============================================================

_CUDA_SRC_INT8_ROWWISE = r"""
extern "C" {

__device__ __forceinline__ float bf16_to_float(unsigned short x) {
    return __int_as_float(((unsigned int)x) << 16);
}

__device__ __forceinline__ unsigned short float_to_bf16_rn_bits(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int lsb = (u >> 16) & 1U;
    unsigned int rounding_bias = 0x7FFFU + lsb;
    return (unsigned short)((u + rounding_bias) >> 16);
}

__device__ __forceinline__ unsigned int f32_to_s8_sat(float x) {
    unsigned int res;
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(x));
    return res & 0xFF;
}

__global__ void __launch_bounds__(1024) kernel_quant_int8_rowwise(
    const unsigned short* __restrict__ x_bf16,
    signed char* __restrict__ q_i8,
    float* __restrict__ scales,
    int rows,
    int cols,
    float pre_scale
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int cols_vec = cols >> 3; 

    long long row_offset = ((long long)row) * cols;
    const uint4* x_row_vec = (const uint4*)(x_bf16 + row_offset);
    uint2* q_row_vec = (uint2*)(q_i8 + row_offset);

    unsigned int local_amax_i = 0;

    for (int c = tid; c < cols_vec; c += blockDim.x) {
        uint4 v = x_row_vec[c];
        
        // Strip sign bit (15th bit of bf16) to find absolute maximums purely via integer math
        unsigned int w0 = v.x & 0x7FFF7FFF;
        unsigned int w1 = v.y & 0x7FFF7FFF;
        unsigned int w2 = v.z & 0x7FFF7FFF;
        unsigned int w3 = v.w & 0x7FFF7FFF;
        
        unsigned int max01 = max(max(w0 & 0xFFFF, w0 >> 16), max(w1 & 0xFFFF, w1 >> 16));
        unsigned int max23 = max(max(w2 & 0xFFFF, w2 >> 16), max(w3 & 0xFFFF, w3 >> 16));
        
        local_amax_i = max(local_amax_i, max(max01, max23));
    }

    const unsigned short* x_row = x_bf16 + row_offset;
    signed char* q_row = q_i8 + row_offset;

    int vec_end = cols_vec << 3;
    for (int c = vec_end + tid; c < cols; c += blockDim.x) {
        unsigned int w = x_row[c] & 0x7FFF;
        local_amax_i = max(local_amax_i, w);
    }

    // Convert the single integer absolute max back to float representation
    float local_amax = __int_as_float(local_amax_i << 16) * pre_scale;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_amax = fmaxf(local_amax, __shfl_down_sync(0xffffffff, local_amax, offset));
    }

    __shared__ float warp_max[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    if (lane == 0) warp_max[warp_id] = local_amax;
    __syncthreads();

    if (warp_id == 0) {
        float block_amax = (lane < num_warps) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_amax = fmaxf(block_amax, __shfl_down_sync(0xffffffff, block_amax, offset));
        }
        if (lane == 0) {
            float scale = (block_amax > 0.0f) ? (block_amax * (1.0f / 127.0f)) : 1.0f;
            scales[row] = scale;
            warp_max[0] = pre_scale / scale; 
        }
    }
    __syncthreads();

    float inv_scale = warp_max[0];

    for (int c = tid; c < cols_vec; c += blockDim.x) {
        uint4 v = x_row_vec[c];
        
        // Fast bf16 extraction avoiding right shifts for the upper bits
        float f0 = __int_as_float((v.x & 0x0000FFFF) << 16) * inv_scale;
        float f1 = __int_as_float(v.x & 0xFFFF0000) * inv_scale;
        float f2 = __int_as_float((v.y & 0x0000FFFF) << 16) * inv_scale;
        float f3 = __int_as_float(v.y & 0xFFFF0000) * inv_scale;
        float f4 = __int_as_float((v.z & 0x0000FFFF) << 16) * inv_scale;
        float f5 = __int_as_float(v.z & 0xFFFF0000) * inv_scale;
        float f6 = __int_as_float((v.w & 0x0000FFFF) << 16) * inv_scale;
        float f7 = __int_as_float(v.w & 0xFFFF0000) * inv_scale;

        unsigned int p0 = f32_to_s8_sat(f0) | (f32_to_s8_sat(f1) << 8) | (f32_to_s8_sat(f2) << 16) | (f32_to_s8_sat(f3) << 24);
        unsigned int p1 = f32_to_s8_sat(f4) | (f32_to_s8_sat(f5) << 8) | (f32_to_s8_sat(f6) << 16) | (f32_to_s8_sat(f7) << 24);

        q_row_vec[c] = make_uint2(p0, p1);
    }

    for (int c = vec_end + tid; c < cols; c += blockDim.x) {
        float f = __int_as_float(((unsigned int)x_row[c]) << 16) * inv_scale;
        q_row[c] = (signed char)f32_to_s8_sat(f);
    }
}

__global__ void __launch_bounds__(1024) kernel_dequant_int8_rowwise(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ scales,
    unsigned short* __restrict__ out_bf16,
    int rows,
    int cols,
    float inv_pre_scale
) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;

    int tid = (int)threadIdx.x;
    int cols_vec = cols >> 3;

    long long row_offset = ((long long)row) * ((long long)cols);
    const uint2* q_row_vec = (const uint2*)(q_i8 + row_offset);
    uint4* out_row_vec = (uint4*)(out_bf16 + row_offset);

    float mul = scales[row] * inv_pre_scale;

    for (int c = tid; c < cols_vec; c += blockDim.x) {
        uint2 p = q_row_vec[c];
        unsigned int p0 = p.x;
        unsigned int p1 = p.y;

        int q0 = (int)(signed char)(p0 & 0xff);
        int q1 = (int)(signed char)((p0 >> 8) & 0xff);
        int q2 = (int)(signed char)((p0 >> 16) & 0xff);
        int q3 = (int)(signed char)(p0 >> 24);
        int q4 = (int)(signed char)(p1 & 0xff);
        int q5 = (int)(signed char)((p1 >> 8) & 0xff);
        int q6 = (int)(signed char)((p1 >> 16) & 0xff);
        int q7 = (int)(signed char)(p1 >> 24);

        unsigned short h0 = float_to_bf16_rn_bits(((float)q0) * mul);
        unsigned short h1 = float_to_bf16_rn_bits(((float)q1) * mul);
        unsigned short h2 = float_to_bf16_rn_bits(((float)q2) * mul);
        unsigned short h3 = float_to_bf16_rn_bits(((float)q3) * mul);
        unsigned short h4 = float_to_bf16_rn_bits(((float)q4) * mul);
        unsigned short h5 = float_to_bf16_rn_bits(((float)q5) * mul);
        unsigned short h6 = float_to_bf16_rn_bits(((float)q6) * mul);
        unsigned short h7 = float_to_bf16_rn_bits(((float)q7) * mul);

        uint4 out;
        out.x = ((unsigned int)h0) | (((unsigned int)h1) << 16);
        out.y = ((unsigned int)h2) | (((unsigned int)h3) << 16);
        out.z = ((unsigned int)h4) | (((unsigned int)h5) << 16);
        out.w = ((unsigned int)h6) | (((unsigned int)h7) << 16);

        out_row_vec[c] = out;
    }

    int vec_end = cols_vec << 3;
    const signed char* q_row = q_i8 + row_offset;
    unsigned short* out_row = out_bf16 + row_offset;
    for (int c = vec_end + tid; c < cols; c += blockDim.x) {
        float v = ((float)q_row[c]) * mul;
        out_row[c] = float_to_bf16_rn_bits(v);
    }
}
}
"""

@dataclass
class _Int8Kernels:
    quant_rowwise: cp.RawKernel
    dequant_rowwise: cp.RawKernel

    @staticmethod
    def build() -> "_Int8Kernels":
        rowwise_module = cp.RawModule(
            code=_CUDA_SRC_INT8_ROWWISE,
            options=("--std=c++17", "--use_fast_math"),
            backend="nvrtc",
        )

        return _Int8Kernels(
            quant_rowwise=rowwise_module.get_function("kernel_quant_int8_rowwise"),
            dequant_rowwise=rowwise_module.get_function("kernel_dequant_int8_rowwise"),
        )


_INT8_KERNELS: Optional[_Int8Kernels] = None


def _get_int8_kernels() -> _Int8Kernels:
    global _INT8_KERNELS
    if _INT8_KERNELS is None:
        _INT8_KERNELS = _Int8Kernels.build()
    return _INT8_KERNELS


def quant_int8(
    x_bf16: torch.Tensor,
    *,
    pre_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x_bf16.contiguous()
    require(
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() >= 1,
        "quant_int8 expects a CUDA BF16 tensor with at least 1 dimension.",
    )
    require(pre_scale > 0.0, "pre_scale must be > 0.")

    original_shape = tuple(x.shape)
    cols = original_shape[-1]
    require(cols > 0, "The last dimension must be non-empty.")

    rows = x.numel() // cols

    x_2d = x.view(rows, cols)
    q_2d = torch.empty((rows, cols), device=x.device, dtype=torch.int8)
    scales_1d = torch.empty((rows,), device=x.device, dtype=torch.float32)

    kernels = _get_int8_kernels()
    threads = _pick_row_threads(cols)
    grid = (rows,)

    kernels.quant_rowwise(
        grid,
        (threads,),
        (
            x_2d.data_ptr(),
            q_2d.data_ptr(),
            scales_1d.data_ptr(),
            rows,
            cols,
            cp.float32(pre_scale),
        ),
    )

    q = q_2d.view(*original_shape)
    scales = scales_1d.view(*original_shape[:-1])
    return q, scales


def dequant_int8(
    q_int8: torch.Tensor,
    scales: torch.Tensor,
    *,
    pre_scale: float = 1.0,
) -> torch.Tensor:
    q = q_int8.contiguous()
    s = scales.contiguous()

    require(
        q.is_cuda and q.dtype == torch.int8 and q.dim() >= 1,
        "dequant_int8 expects a CUDA int8 tensor with at least 1 dimension.",
    )
    require(
        s.is_cuda and s.dtype == torch.float32,
        "dequant_int8 expects CUDA float32 scales.",
    )
    require(pre_scale > 0.0, "pre_scale must be > 0.")

    original_shape = tuple(q.shape)
    cols = original_shape[-1]
    require(cols > 0, "The last dimension must be non-empty.")

    expected_scale_shape = original_shape[:-1]
    require(
        tuple(s.shape) == expected_scale_shape,
        f"scales must have shape {expected_scale_shape}, got {tuple(s.shape)}.",
    )

    rows = q.numel() // cols
    q_2d = q.view(rows, cols)
    scales_1d = s.view(rows)
    out_2d = torch.empty((rows, cols), device=q.device, dtype=torch.bfloat16)

    kernels = _get_int8_kernels()
    threads = _pick_row_threads(cols)
    grid = (rows,)
    
    kernels.dequant_rowwise(
        grid,
        (threads,),
        (
            q_2d.data_ptr(),
            scales_1d.data_ptr(),
            out_2d.data_ptr(),
            rows,
            cols,
            cp.float32(1.0 / pre_scale),
        ),
    )

    return out_2d.view(*original_shape)


# ============================================================
# MXFP8
# ============================================================

_CUDA_SRC_MXFP8_QUANT = r"""
extern "C" {

__device__ __forceinline__ unsigned char cvt_e4m3(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}

__device__ __forceinline__ int exp_unbiased_rceil_448(float amax) {
    if (!(amax > 0.0f)) return 0;
    int bits = __float_as_int(amax);
    int biased_e = (bits >> 23) & 0xff;
    int e = biased_e - 127;
    int mant_bits = (bits & 0x007fffff) | 0x3f800000;
    float m = __int_as_float(mant_bits);
    int exp = (e - 8) + ((m > 1.75f) ? 1 : 0);
    return exp;
}

__device__ __forceinline__ unsigned char exp_to_e8m0_biased(int exp_unbiased) {
    int e = exp_unbiased + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

__device__ __forceinline__ float inv_scale_from_exp(int exp_unbiased) {
    int be = 127 - exp_unbiased;
    if (be <= 0) return 0.0f;
    if (be >= 255) return __int_as_float(0x7f800000);
    return __int_as_float((be & 0xff) << 23);
}

__global__ void kernel_quant_mxfp8(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_fp8,
    unsigned char* __restrict__ s_fp8,
    int num_elements,
    int scale_cols
){
    int num_blocks32 = num_elements >> 5;

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = tid & 31;
    int warp_id = tid >> 5;

    int block_base = warp_id << 4;
    int block_in_warp = lane >> 1;
    int block_idx = block_base + block_in_warp;
    if (block_idx >= num_blocks32) return;

    int half = lane & 1;
    int elem_base = (block_idx << 5) + (half << 4);

    const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
    uint4 v0 = in4[0];
    uint4 v1 = in4[1];

    unsigned int w[8];
    w[0] = v0.x; w[1] = v0.y; w[2] = v0.z; w[3] = v0.w;
    w[4] = v1.x; w[5] = v1.y; w[6] = v1.z; w[7] = v1.w;

    float f[16];
    float local_max = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned int u = w[i];
        unsigned short lo = (unsigned short)(u & 0xffff);
        unsigned short hi = (unsigned short)(u >> 16);

        float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
        float a1 = __int_as_float((int)(((unsigned int)hi) << 16));

        f[2 * i + 0] = a0;
        f[2 * i + 1] = a1;

        local_max = fmaxf(local_max, fabsf(a0));
        local_max = fmaxf(local_max, fabsf(a1));
    }

    float other = __shfl_xor_sync(0xffffffff, local_max, 1);
    float amax = fmaxf(local_max, other);

    int exp_unbiased = exp_unbiased_rceil_448(amax);
    unsigned char e8 = (amax > 0.0f) ? exp_to_e8m0_biased(exp_unbiased) : (unsigned char)0;
    float inv_scale = (amax > 0.0f) ? inv_scale_from_exp(exp_unbiased) : 1.0f;

    if (half == 0) {
        int r = block_idx / scale_cols;
        int c = block_idx % scale_cols;

        int r_tile = r >> 7;
        int c_tile = c >> 2;
        int r_in_tile = r & 127;
        int c_in_tile = c & 3;

        int scale_cols_div_4 = scale_cols >> 2;
        int tile_idx = r_tile * scale_cols_div_4 + c_tile;

        int swizzled_row = r_in_tile & 31;
        int swizzled_col = ((r_in_tile >> 5) << 2) + c_in_tile;

        int swizzled_idx = (tile_idx << 9) + (swizzled_row << 4) + swizzled_col;
        s_fp8[swizzled_idx] = e8;
    }

    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        unsigned char q = cvt_e4m3(f[i] * inv_scale);
        int sh = (i & 3) * 8;
        if (i < 4)       p0 |= ((unsigned int)q) << sh;
        else if (i < 8)  p1 |= ((unsigned int)q) << sh;
        else if (i < 12) p2 |= ((unsigned int)q) << sh;
        else             p3 |= ((unsigned int)q) << sh;
    }

    uint4 out;
    out.x = p0;
    out.y = p1;
    out.z = p2;
    out.w = p3;
    ((uint4*)(q_fp8 + elem_base))[0] = out;
}
}
"""

_CUDA_SRC_MXFP8_RMSNORM_QUANT = r"""
extern "C" {

__device__ __forceinline__ unsigned char cvt_e4m3(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}

__device__ __forceinline__ int exp_unbiased_rceil_448(float amax) {
    if (!(amax > 0.0f)) return 0;
    int bits = __float_as_int(amax);
    int biased_e = (bits >> 23) & 0xff;
    int e = biased_e - 127;
    int mant_bits = (bits & 0x007fffff) | 0x3f800000;
    float m = __int_as_float(mant_bits);
    int exp = (e - 8) + ((m > 1.75f) ? 1 : 0);
    return exp;
}

__device__ __forceinline__ unsigned char exp_to_e8m0_biased(int exp_unbiased) {
    int e = exp_unbiased + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

__device__ __forceinline__ float inv_scale_from_exp(int exp_unbiased) {
    int be = 127 - exp_unbiased;
    if (be <= 0) return 0.0f;
    if (be >= 255) return __int_as_float(0x7f800000);
    return __int_as_float((be & 0xff) << 23);
}

__device__ __forceinline__ unsigned short float_to_bf16_rn_bits(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int lsb = (u >> 16) & 1U;
    unsigned int rounding_bias = 0x7FFFU + lsb;
    return (unsigned short)((u + rounding_bias) >> 16);
}

__global__ void kernel_rmsnorm_quant_mxfp8(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_fp8,
    unsigned char* __restrict__ s_fp8,
    unsigned short* __restrict__ inv_rms_bf16,
    int rows,
    int cols,
    int padded_scale_cols,
    float epsilon
){
    extern __shared__ unsigned short smem_row[];
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int cols16 = cols / 16;

    float sum_sq = 0.0f;
    for (int c = tid; c < cols16; c += blockDim.x) {
        int elem_base = row * cols + (c * 16);
        const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
        uint4 v0 = in4[0];
        uint4 v1 = in4[1];

        uint4* smem_out = (uint4*)(smem_row + (c * 16));
        smem_out[0] = v0;
        smem_out[1] = v1;

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);
            float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16));
            sum_sq += a0 * a0 + a1 * a1;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    static __shared__ float shared_sum[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;

    if (lane == 0) shared_sum[warp_id] = sum_sq;
    __syncthreads();

    float block_sum = (tid < (blockDim.x >> 5)) ? shared_sum[lane] : 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (tid == 0) {
            float inv_rms = rsqrtf((block_sum / (float)cols) + epsilon);
            shared_sum[0] = inv_rms;
            inv_rms_bf16[row] = float_to_bf16_rn_bits(inv_rms);
        }
    }
    __syncthreads();

    float inv_rms = shared_sum[0];
    int scale_cols = cols / 32;

    for (int block_idx = tid >> 1; block_idx < scale_cols; block_idx += (blockDim.x >> 1)) {
        int half = tid & 1;

        int smem_idx = (block_idx * 32) + (half * 16);
        const uint4* in4 = (const uint4*)(smem_row + smem_idx);
        uint4 v0 = in4[0];
        uint4 v1 = in4[1];

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float f[16];
        float local_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);

            float a0 = __int_as_float((int)(((unsigned int)lo) << 16)) * inv_rms;
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16)) * inv_rms;

            f[2 * i + 0] = a0;
            f[2 * i + 1] = a1;
            local_max = fmaxf(local_max, fabsf(a0));
            local_max = fmaxf(local_max, fabsf(a1));
        }

        float other = __shfl_xor_sync(0xffffffff, local_max, 1);
        float amax = fmaxf(local_max, other);

        int exp_unbiased = exp_unbiased_rceil_448(amax);
        unsigned char e8 = (amax > 0.0f) ? exp_to_e8m0_biased(exp_unbiased) : (unsigned char)0;
        float inv_scale = (amax > 0.0f) ? inv_scale_from_exp(exp_unbiased) : 1.0f;

        if (half == 0) {
            int r_tile = row >> 7;
            int c_tile = block_idx >> 2;
            int r_in_tile = row & 127;
            int c_in_tile = block_idx & 3;

            int tile_idx = r_tile * (padded_scale_cols >> 2) + c_tile;

            int swizzled_row = r_in_tile & 31;
            int swizzled_col = ((r_in_tile >> 5) << 2) + c_in_tile;

            int swizzled_idx = (tile_idx << 9) + (swizzled_row << 4) + swizzled_col;
            s_fp8[swizzled_idx] = e8;
        }

        unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char q = cvt_e4m3(f[i] * inv_scale);
            int sh = (i & 3) * 8;
            if (i < 4)       p0 |= ((unsigned int)q) << sh;
            else if (i < 8)  p1 |= ((unsigned int)q) << sh;
            else if (i < 12) p2 |= ((unsigned int)q) << sh;
            else             p3 |= ((unsigned int)q) << sh;
        }

        uint4 out;
        out.x = p0;
        out.y = p1;
        out.z = p2;
        out.w = p3;

        int global_q_idx = row * cols + smem_idx;
        ((uint4*)(q_fp8 + global_q_idx))[0] = out;
    }
}
}
"""

_CUDA_SRC_MXFP8_DEQUANT = r"""
extern "C" {

__device__ __forceinline__ float cvt_e4m3_to_f32(unsigned char x) {
    if ((x & 0x7f) == 0) return 0.0f;
    if ((x & 0x7f) == 0x7f) return __uint_as_float(((x & 0x80) << 24) | 0x7fc00000);

    unsigned int sign = (x & 0x80) << 24;
    unsigned int exp  = (x & 0x78) >> 3;
    unsigned int mant = x & 0x07;

    if (exp == 0) {
        float val = (float)mant * 0.001953125f;
        return (x & 0x80) ? -val : val;
    }

    exp += 120;
    unsigned int bits = sign | (exp << 23) | (mant << 20);
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short float_to_bf16_rn_bits(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int lsb = (u >> 16) & 1U;
    unsigned int rounding_bias = 0x7FFFU + lsb;
    return (unsigned short)((u + rounding_bias) >> 16);
}

__global__ void kernel_dequant_mxfp8(
    const unsigned char* __restrict__ q_fp8,
    const unsigned char* __restrict__ s_fp8,
    unsigned short* __restrict__ out_bf16,
    int num_elements,
    int scale_cols
){
    int num_blocks32 = num_elements >> 5;

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = tid & 31;
    int warp_id = tid >> 5;

    int block_base = warp_id << 4;
    int block_in_warp = lane >> 1;
    int block_idx = block_base + block_in_warp;
    if (block_idx >= num_blocks32) return;

    int half = lane & 1;
    int elem_base = (block_idx << 5) + (half << 4);

    int r = block_idx / scale_cols;
    int c = block_idx % scale_cols;

    int r_tile = r >> 7;
    int c_tile = c >> 2;
    int r_in_tile = r & 127;
    int c_in_tile = c & 3;

    int scale_cols_div_4 = scale_cols >> 2;
    int tile_idx = r_tile * scale_cols_div_4 + c_tile;

    int swizzled_row = r_in_tile & 31;
    int swizzled_col = ((r_in_tile >> 5) << 2) + c_in_tile;

    int swizzled_idx = (tile_idx << 9) + (swizzled_row << 4) + swizzled_col;

    unsigned char e8 = s_fp8[swizzled_idx];
    float scale = (e8 > 0) ? __uint_as_float(((unsigned int)e8) << 23) : 0.0f;

    const uint4* in_q = (const uint4*)(q_fp8 + elem_base);
    uint4 q_v = in_q[0];

    unsigned int p0 = q_v.x;
    unsigned int p1 = q_v.y;
    unsigned int p2 = q_v.z;
    unsigned int p3 = q_v.w;

    unsigned short out[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        unsigned int p = (i < 4) ? p0 : ((i < 8) ? p1 : ((i < 12) ? p2 : p3));
        unsigned char q = (unsigned char)((p >> ((i & 3) * 8)) & 0xff);

        float f = cvt_e4m3_to_f32(q) * scale;
        out[i] = float_to_bf16_rn_bits(f);
    }

    uint4 out0, out1;
    out0.x = ((unsigned int)out[0])  | (((unsigned int)out[1]) << 16);
    out0.y = ((unsigned int)out[2])  | (((unsigned int)out[3]) << 16);
    out0.z = ((unsigned int)out[4])  | (((unsigned int)out[5]) << 16);
    out0.w = ((unsigned int)out[6])  | (((unsigned int)out[7]) << 16);

    out1.x = ((unsigned int)out[8])  | (((unsigned int)out[9]) << 16);
    out1.y = ((unsigned int)out[10]) | (((unsigned int)out[11]) << 16);
    out1.z = ((unsigned int)out[12]) | (((unsigned int)out[13]) << 16);
    out1.w = ((unsigned int)out[14]) | (((unsigned int)out[15]) << 16);

    uint4* out_bf16_ptr = (uint4*)(out_bf16 + elem_base);
    out_bf16_ptr[0] = out0;
    out_bf16_ptr[1] = out1;
}
}
"""


@dataclass
class _MXFP8Kernels:
    quant: cp.RawKernel
    rmsnorm_quant: cp.RawKernel
    dequant: cp.RawKernel

    @staticmethod
    def build() -> "_MXFP8Kernels":
        quant_module = cp.RawModule(code=_CUDA_SRC_MXFP8_QUANT, options=("--std=c++17",))
        rmsnorm_quant_module = cp.RawModule(
            code=_CUDA_SRC_MXFP8_RMSNORM_QUANT,
            options=("--std=c++17",),
        )
        dequant_module = cp.RawModule(code=_CUDA_SRC_MXFP8_DEQUANT, options=("--std=c++17",))

        rmsnorm_quant_kernel = rmsnorm_quant_module.get_function("kernel_rmsnorm_quant_mxfp8")
        rmsnorm_quant_kernel.max_dynamic_shared_size_bytes = 98304

        return _MXFP8Kernels(
            quant=quant_module.get_function("kernel_quant_mxfp8"),
            rmsnorm_quant=rmsnorm_quant_kernel,
            dequant=dequant_module.get_function("kernel_dequant_mxfp8"),
        )


_MXFP8_KERNELS: Optional[_MXFP8Kernels] = None


def _get_mxfp8_kernels() -> _MXFP8Kernels:
    global _MXFP8_KERNELS
    if _MXFP8_KERNELS is None:
        _MXFP8_KERNELS = _MXFP8Kernels.build()
    return _MXFP8_KERNELS


def quant_mxfp8(
    x_bf16: torch.Tensor,
    *,
    apply_rmsnorm: bool = False,
    epsilon: float = 1e-6,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    x = x_bf16.contiguous()
    require(
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2,
        "quant_mxfp8 expects a CUDA BF16 2D tensor.",
    )

    rows, cols = x.shape
    require(cols % 32 == 0, "K must be divisible by 32 for MXFP8 1x32 scaling.")

    num_elements = x.numel()
    scale_cols = cols // 32

    padded_rows = ((rows + 127) // 128) * 128
    padded_scale_cols = ((scale_cols + 3) // 4) * 4

    q = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((padded_rows, padded_scale_cols), device=x.device, dtype=torch.float8_e8m0fnu)

    kernels = _get_mxfp8_kernels()

    if not apply_rmsnorm:
        threads = 256
        warps_per_block = threads // 32
        blocks_per_warp = 16

        num_blocks32 = num_elements // 32
        num_warps = (num_blocks32 + blocks_per_warp - 1) // blocks_per_warp
        grid = ((num_warps + warps_per_block - 1) // warps_per_block,)

        kernels.quant(
            grid,
            (threads,),
            (
                x.data_ptr(),
                q.data_ptr(),
                s.data_ptr(),
                num_elements,
                scale_cols,
            ),
        )
        return q, s

    inv_rms = torch.empty((rows,), device=x.device, dtype=torch.bfloat16)

    threads = 256
    grid = (rows,)
    shared_mem_bytes = cols * 2

    kernels.rmsnorm_quant(
        grid,
        (threads,),
        (
            x.data_ptr(),
            q.data_ptr(),
            s.data_ptr(),
            inv_rms.data_ptr(),
            rows,
            cols,
            padded_scale_cols,
            cp.float32(epsilon),
        ),
        shared_mem=shared_mem_bytes,
    )

    return q, s, inv_rms


def dequant_mxfp8(
    q_fp8: torch.Tensor,
    s_fp8: torch.Tensor,
) -> torch.Tensor:
    q = q_fp8.contiguous()
    s = s_fp8.contiguous()

    require(
        q.is_cuda and q.dtype == torch.float8_e4m3fn and q.dim() == 2,
        "dequant_mxfp8 expects a CUDA E4M3 2D tensor for the quantized inputs.",
    )
    require(
        s.is_cuda and s.dtype == torch.float8_e8m0fnu and s.dim() == 2,
        "dequant_mxfp8 expects a CUDA E8M0 2D tensor for the scales.",
    )

    rows, cols = q.shape
    require(cols % 32 == 0, "K must be divisible by 32 for MXFP8 1x32 scaling.")

    num_elements = q.numel()
    scale_cols = cols // 32

    out = torch.empty((rows, cols), device=q.device, dtype=torch.bfloat16)

    threads = 256
    warps_per_block = threads // 32
    blocks_per_warp = 16

    num_blocks32 = num_elements // 32
    num_warps = (num_blocks32 + blocks_per_warp - 1) // blocks_per_warp
    grid = ((num_warps + warps_per_block - 1) // warps_per_block,)

    _get_mxfp8_kernels().dequant(
        grid,
        (threads,),
        (
            q.data_ptr(),
            s.data_ptr(),
            out.data_ptr(),
            num_elements,
            scale_cols,
        ),
    )

    return out


# ============================================================
# NVFP4
# ============================================================

_CUDA_SRC_NVFP4_QUANT = r"""
extern "C" {

__device__ __forceinline__ unsigned char fp8_e4m3_from_f32(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}

__device__ __forceinline__ float f32_from_fp8_e4m3(unsigned char b) {
    unsigned short e4m3x2 = (unsigned short)b | ((unsigned short)b << 8);
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(e4m3x2));
    unsigned short h0 = (unsigned short)(f16x2 & 0xFFFF);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(h0));
    return out;
}

__global__ void kernel_quant_nvfp4(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_u8,
    unsigned char* __restrict__ s_u8,
    int num_blocks16,
    int scale_cols,
    int padded_scale_cols
){
    int block_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block_idx >= num_blocks16) return;

    int elem_base = block_idx << 4;

    const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
    uint4 v0 = in4[0];
    uint4 v1 = in4[1];

    unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float f[16];
    float amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned short lo = (unsigned short)(w[i] & 0xffff);
        unsigned short hi = (unsigned short)(w[i] >> 16);
        float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
        float a1 = __int_as_float((int)(((unsigned int)hi) << 16));
        f[2 * i + 0] = a0;
        f[2 * i + 1] = a1;
        amax = fmaxf(amax, fabsf(a0));
        amax = fmaxf(amax, fabsf(a1));
    }

    unsigned char scale_b = 0;
    float inv_scale = 1.0f;
    if (amax > 0.0f) {
        float scale_f = amax * (1.0f / 6.0f);
        scale_b = fp8_e4m3_from_f32(scale_f);
        float scale_fq = f32_from_fp8_e4m3(scale_b);
        inv_scale = (scale_fq > 0.0f) ? (1.0f / scale_fq) : 1.0f;
    }

    int r = block_idx / scale_cols;
    int c = block_idx % scale_cols;
    int r_tile = r >> 7;
    int c_tile = c >> 2;
    int r_in_tile = r & 127;
    int c_in_tile = c & 3;
    int tile_idx = r_tile * (padded_scale_cols >> 2) + c_tile;
    int swizzled_row = r_in_tile & 31;
    int swizzled_col = ((r_in_tile >> 5) << 2) + c_in_tile;
    int swizzled_idx = (tile_idx << 9) + (swizzled_row << 4) + swizzled_col;
    s_u8[swizzled_idx] = scale_b;

    unsigned int out0 = 0;
    unsigned int out1 = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float ax0 = fabsf(f[2 * i] * inv_scale);
        float ax1 = fabsf(f[2 * i + 1] * inv_scale);
        if (ax0 > 6.0f) ax0 = 6.0f;
        if (ax1 > 6.0f) ax1 = 6.0f;

        int sign0 = f[2 * i] < 0.0f;
        int sign1 = f[2 * i + 1] < 0.0f;

        unsigned int code0 =
            (!(ax0 < 0.25f)) +
            (!(ax0 < 0.75f)) +
            (!(ax0 < 1.25f)) +
            (!(ax0 < 1.75f)) +
            (!(ax0 < 2.5f)) +
            (!(ax0 < 3.5f)) +
            (!(ax0 < 5.0f));

        unsigned int code1 =
            (!(ax1 < 0.25f)) +
            (!(ax1 < 0.75f)) +
            (!(ax1 < 1.25f)) +
            (!(ax1 < 1.75f)) +
            (!(ax1 < 2.5f)) +
            (!(ax1 < 3.5f)) +
            (!(ax1 < 5.0f));

        unsigned char n0 = (unsigned char)((sign0 << 3) | code0);
        unsigned char n1 = (unsigned char)((sign1 << 3) | code1);
        unsigned char packed = (unsigned char)((n0 & 0xF) | ((n1 & 0xF) << 4));

        if (i < 4) out0 |= ((unsigned int)packed) << (i * 8);
        else       out1 |= ((unsigned int)packed) << ((i - 4) * 8);
    }

    ((uint2*)(q_u8 + (block_idx << 3)))[0] = make_uint2(out0, out1);
}
}
"""

_CUDA_SRC_NVFP4_RMSNORM_QUANT = r"""
extern "C" {

__device__ __forceinline__ unsigned char fp8_e4m3_from_f32(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}

__device__ __forceinline__ float f32_from_fp8_e4m3(unsigned char b) {
    unsigned short e4m3x2 = (unsigned short)b | ((unsigned short)b << 8);
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(e4m3x2));
    unsigned short h0 = (unsigned short)(f16x2 & 0xFFFF);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(h0));
    return out;
}

__device__ __forceinline__ unsigned short float_to_bf16_rn_bits(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int lsb = (u >> 16) & 1U;
    unsigned int rounding_bias = 0x7FFFU + lsb;
    return (unsigned short)((u + rounding_bias) >> 16);
}

__global__ void kernel_rmsnorm_quant_nvfp4(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_u8,
    unsigned char* __restrict__ s_u8,
    unsigned short* __restrict__ inv_rms_bf16,
    int rows,
    int cols,
    int padded_scale_cols,
    float epsilon
){
    extern __shared__ unsigned short smem_row[];
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int scale_cols = cols / 16;

    float sum_sq = 0.0f;
    for (int c = tid; c < scale_cols; c += blockDim.x) {
        int elem_base = row * cols + (c * 16);
        const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
        uint4 v0 = in4[0];
        uint4 v1 = in4[1];

        uint4* smem_out = (uint4*)(smem_row + (c * 16));
        smem_out[0] = v0;
        smem_out[1] = v1;

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);
            float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16));
            sum_sq += a0 * a0 + a1 * a1;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    static __shared__ float shared_sum[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;

    if (lane == 0) shared_sum[warp_id] = sum_sq;
    __syncthreads();

    float block_sum = (tid < (blockDim.x >> 5)) ? shared_sum[lane] : 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (tid == 0) {
            float inv_rms = rsqrtf((block_sum / (float)cols) + epsilon);
            shared_sum[0] = inv_rms;
            inv_rms_bf16[row] = float_to_bf16_rn_bits(inv_rms);
        }
    }
    __syncthreads();

    float inv_rms = shared_sum[0];

    for (int c = tid; c < scale_cols; c += blockDim.x) {
        int block_idx = row * scale_cols + c;
        const uint4* in4 = (const uint4*)(smem_row + (c * 16));
        uint4 v0 = in4[0];
        uint4 v1 = in4[1];

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float f[16];
        float amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);

            float a0 = __int_as_float((int)(((unsigned int)lo) << 16)) * inv_rms;
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16)) * inv_rms;

            f[2 * i + 0] = a0;
            f[2 * i + 1] = a1;
            amax = fmaxf(amax, fabsf(a0));
            amax = fmaxf(amax, fabsf(a1));
        }

        unsigned char scale_b = 0;
        float inv_scale = 1.0f;
        if (amax > 0.0f) {
            float scale_f = amax * (1.0f / 6.0f);
            scale_b = fp8_e4m3_from_f32(scale_f);
            float scale_fq = f32_from_fp8_e4m3(scale_b);
            inv_scale = (scale_fq > 0.0f) ? (1.0f / scale_fq) : 1.0f;
        }

        int r_tile = row >> 7;
        int c_tile = c >> 2;
        int r_in_tile = row & 127;
        int c_in_tile = c & 3;
        int tile_idx = r_tile * (padded_scale_cols >> 2) + c_tile;
        int swizzled_row = r_in_tile & 31;
        int swizzled_col = ((r_in_tile >> 5) << 2) + c_in_tile;
        int swizzled_idx = (tile_idx << 9) + (swizzled_row << 4) + swizzled_col;
        s_u8[swizzled_idx] = scale_b;

        unsigned int out0 = 0;
        unsigned int out1 = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float ax0 = fabsf(f[2 * i] * inv_scale);
            float ax1 = fabsf(f[2 * i + 1] * inv_scale);
            if (ax0 > 6.0f) ax0 = 6.0f;
            if (ax1 > 6.0f) ax1 = 6.0f;

            int sign0 = f[2 * i] < 0.0f;
            int sign1 = f[2 * i + 1] < 0.0f;

            unsigned int code0 =
                (!(ax0 < 0.25f)) +
                (!(ax0 < 0.75f)) +
                (!(ax0 < 1.25f)) +
                (!(ax0 < 1.75f)) +
                (!(ax0 < 2.5f)) +
                (!(ax0 < 3.5f)) +
                (!(ax0 < 5.0f));

            unsigned int code1 =
                (!(ax1 < 0.25f)) +
                (!(ax1 < 0.75f)) +
                (!(ax1 < 1.25f)) +
                (!(ax1 < 1.75f)) +
                (!(ax1 < 2.5f)) +
                (!(ax1 < 3.5f)) +
                (!(ax1 < 5.0f));

            unsigned char n0 = (unsigned char)((sign0 << 3) | code0);
            unsigned char n1 = (unsigned char)((sign1 << 3) | code1);
            unsigned char packed = (unsigned char)((n0 & 0xF) | ((n1 & 0xF) << 4));

            if (i < 4) out0 |= ((unsigned int)packed) << (i * 8);
            else       out1 |= ((unsigned int)packed) << ((i - 4) * 8);
        }

        ((uint2*)(q_u8 + (block_idx << 3)))[0] = make_uint2(out0, out1);
    }
}
}
"""

_CUDA_SRC_NVFP4_DEQUANT = r"""
extern "C" {

__device__ __forceinline__ float f32_from_fp8_e4m3(unsigned char b) {
    if (b == 0) return 0.0f;
    unsigned short e4m3x2 = (unsigned short)b | ((unsigned short)b << 8);
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(e4m3x2));
    unsigned short h0 = (unsigned short)(f16x2 & 0xFFFF);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(h0));
    return out;
}

__device__ __forceinline__ float fp4_e2m1_to_f32(unsigned char x) {
    unsigned int sign = (x >> 3) & 0x1;
    unsigned int code = x & 0x7;

    float mag;
    switch (code) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return sign ? -mag : mag;
}

__device__ __forceinline__ unsigned short float_to_bf16_rn_bits(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int lsb = (u >> 16) & 1U;
    unsigned int rounding_bias = 0x7FFFU + lsb;
    return (unsigned short)((u + rounding_bias) >> 16);
}

__global__ void kernel_dequant_nvfp4(
    const unsigned char* __restrict__ q_u8,
    const unsigned char* __restrict__ s_u8,
    unsigned short* __restrict__ out_bf16,
    int num_blocks16,
    int scale_cols,
    int padded_scale_cols
){
    int block_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block_idx >= num_blocks16) return;

    int r = block_idx / scale_cols;
    int c = block_idx % scale_cols;

    int r_tile = r >> 7;
    int c_tile = c >> 2;
    int r_in_tile = r & 127;
    int c_in_tile = c & 3;
    int tile_idx = r_tile * (padded_scale_cols >> 2) + c_tile;
    int swizzled_row = r_in_tile & 31;
    int swizzled_col = ((r_in_tile >> 5) << 2) + c_in_tile;
    int swizzled_idx = (tile_idx << 9) + (swizzled_row << 4) + swizzled_col;

    unsigned char scale_b = s_u8[swizzled_idx];
    float scale = f32_from_fp8_e4m3(scale_b);

    uint2 packed = ((const uint2*)(q_u8 + (block_idx << 3)))[0];
    unsigned int p0 = packed.x;
    unsigned int p1 = packed.y;

    unsigned short out[16];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned int p = (i < 4) ? p0 : p1;
        unsigned char byte = (unsigned char)((p >> ((i & 3) * 8)) & 0xff);

        unsigned char lo = (unsigned char)(byte & 0x0f);
        unsigned char hi = (unsigned char)((byte >> 4) & 0x0f);

        float f0 = fp4_e2m1_to_f32(lo) * scale;
        float f1 = fp4_e2m1_to_f32(hi) * scale;

        out[2 * i + 0] = float_to_bf16_rn_bits(f0);
        out[2 * i + 1] = float_to_bf16_rn_bits(f1);
    }

    int elem_base = block_idx << 4;

    uint4 out0, out1;
    out0.x = ((unsigned int)out[0])  | (((unsigned int)out[1]) << 16);
    out0.y = ((unsigned int)out[2])  | (((unsigned int)out[3]) << 16);
    out0.z = ((unsigned int)out[4])  | (((unsigned int)out[5]) << 16);
    out0.w = ((unsigned int)out[6])  | (((unsigned int)out[7]) << 16);

    out1.x = ((unsigned int)out[8])  | (((unsigned int)out[9]) << 16);
    out1.y = ((unsigned int)out[10]) | (((unsigned int)out[11]) << 16);
    out1.z = ((unsigned int)out[12]) | (((unsigned int)out[13]) << 16);
    out1.w = ((unsigned int)out[14]) | (((unsigned int)out[15]) << 16);

    uint4* out_ptr = (uint4*)(out_bf16 + elem_base);
    out_ptr[0] = out0;
    out_ptr[1] = out1;
}
}
"""


@dataclass
class _NVFP4Kernels:
    quant: cp.RawKernel
    rmsnorm_quant: cp.RawKernel
    dequant: cp.RawKernel

    @staticmethod
    def build() -> "_NVFP4Kernels":
        quant_module = cp.RawModule(code=_CUDA_SRC_NVFP4_QUANT, options=("--std=c++17",))
        rmsnorm_quant_module = cp.RawModule(
            code=_CUDA_SRC_NVFP4_RMSNORM_QUANT,
            options=("--std=c++17",),
        )
        dequant_module = cp.RawModule(code=_CUDA_SRC_NVFP4_DEQUANT, options=("--std=c++17",))

        rmsnorm_quant_kernel = rmsnorm_quant_module.get_function("kernel_rmsnorm_quant_nvfp4")
        rmsnorm_quant_kernel.max_dynamic_shared_size_bytes = 98304

        return _NVFP4Kernels(
            quant=quant_module.get_function("kernel_quant_nvfp4"),
            rmsnorm_quant=rmsnorm_quant_kernel,
            dequant=dequant_module.get_function("kernel_dequant_nvfp4"),
        )


_NVFP4_KERNELS: Optional[_NVFP4Kernels] = None


def _get_nvfp4_kernels() -> _NVFP4Kernels:
    global _NVFP4_KERNELS
    if _NVFP4_KERNELS is None:
        _NVFP4_KERNELS = _NVFP4Kernels.build()
    return _NVFP4_KERNELS


def quant_nvfp4(
    x_bf16: torch.Tensor,
    *,
    apply_rmsnorm: bool = False,
    epsilon: float = 1e-6,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    x = x_bf16.contiguous()
    require(
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2,
        "quant_nvfp4 expects a CUDA BF16 2D tensor.",
    )

    rows, cols = x.shape
    require(cols % 16 == 0, "K must be divisible by 16 for NVFP4 1x16 scaling.")

    scale_cols = cols // 16
    num_blocks16 = rows * scale_cols

    padded_rows = ((rows + 127) // 128) * 128
    padded_scale_cols = ((scale_cols + 3) // 4) * 4

    q = torch.empty((rows, cols // 2), device=x.device, dtype=FP4_DTYPE)
    s = torch.zeros((padded_rows, padded_scale_cols), device=x.device, dtype=FP8_SCALE_DTYPE)

    kernels = _get_nvfp4_kernels()

    if not apply_rmsnorm:
        threads = 256
        grid = ((num_blocks16 + threads - 1) // threads,)

        kernels.quant(
            grid,
            (threads,),
            (
                x.data_ptr(),
                q.data_ptr(),
                s.data_ptr(),
                num_blocks16,
                scale_cols,
                padded_scale_cols,
            ),
        )
        return q, s

    inv_rms = torch.empty((rows,), device=x.device, dtype=torch.bfloat16)

    threads = 256
    grid = (rows,)
    shared_mem_bytes = cols * 2

    kernels.rmsnorm_quant(
        grid,
        (threads,),
        (
            x.data_ptr(),
            q.data_ptr(),
            s.data_ptr(),
            inv_rms.data_ptr(),
            rows,
            cols,
            padded_scale_cols,
            cp.float32(epsilon),
        ),
        shared_mem=shared_mem_bytes,
    )

    return q, s, inv_rms


def dequant_nvfp4(
    q_fp4: torch.Tensor,
    s_fp8: torch.Tensor,
) -> torch.Tensor:
    q = q_fp4.contiguous()
    s = s_fp8.contiguous()

    require(
        q.is_cuda and q.dtype == FP4_DTYPE and q.dim() == 2,
        "dequant_nvfp4 expects a CUDA FP4 2D tensor for the quantized inputs.",
    )
    require(
        s.is_cuda and s.dtype == FP8_SCALE_DTYPE and s.dim() == 2,
        "dequant_nvfp4 expects a CUDA E4M3 2D tensor for the scales.",
    )

    rows, packed_cols = q.shape
    cols = packed_cols * 2
    require(cols % 16 == 0, "K must be divisible by 16 for NVFP4 1x16 scaling.")

    scale_cols = cols // 16
    padded_rows = ((rows + 127) // 128) * 128
    padded_scale_cols = ((scale_cols + 3) // 4) * 4

    require(
        s.shape[0] >= padded_rows and s.shape[1] >= padded_scale_cols,
        f"s_fp8 must have shape at least ({padded_rows}, {padded_scale_cols}) for q_fp4 shape ({rows}, {packed_cols}).",
    )

    num_blocks16 = rows * scale_cols
    out = torch.empty((rows, cols), device=q.device, dtype=torch.bfloat16)

    threads = 256
    grid = ((num_blocks16 + threads - 1) // threads,)

    _get_nvfp4_kernels().dequant(
        grid,
        (threads,),
        (
            q.data_ptr(),
            s.data_ptr(),
            out.data_ptr(),
            num_blocks16,
            scale_cols,
            s.shape[1],
        ),
    )

    return out


__all__ = [
    "require",
    "quant_int8",
    "dequant_int8",
    "quant_mxfp8",
    "dequant_mxfp8",
    "quant_nvfp4",
    "dequant_nvfp4",
]
