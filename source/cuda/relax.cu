//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/multi_precision.cuh"

surface<void, cudaSurfaceType3D> surf;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_packed;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_u;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_b;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf_u;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf_b;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd_u;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd_b;

const float kBeta  = 6.0f;

template <typename FPType>
struct UpperBoundaryHandlerNeumann
{
    __device__ void HandleUpperBoundary(FPType* north, FPType center, int y,
                                        int height)
    {
    }
};

template <typename FPType>
struct UpperBoundaryHandlerOutflow
{
    __device__ void HandleUpperBoundary(FPType* north, FPType center, int y,
                                        int height)
    {
        if (y == height - 1) {
            if (center > 0.0f)
                *north = -center;
            else
                *north = 0.0f;
        }
    }
};

template <typename FPType>
__device__ void ModifyBoundaryCoef(FPType* coef, uint x, uint y, uint z,
                                   const uint3& volume_size)
{
    if (x == 0)
        *coef -= 1.0f;

    if (y == 0)
        *coef -= 1.0f;

    if (z == 0)
        *coef -= 1.0f;

    if (x == volume_size.x - 1)
        *coef -= 1.0f;

    if (y == volume_size.y - 1)
        *coef -= 1.0f;

    if (z == volume_size.z - 1)
        *coef -= 1.0f;
}

// =============================================================================

template <typename StorageType, typename UpperBoundaryHandler>
__global__ void DampedJacobiKernel(float omega, uint3 volume_size,
                                   UpperBoundaryHandler handler)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    Tex3d<StorageType> t3d;
    FPType near =   t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y,        z - 1.0f);
    FPType south =  t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y - 1.0f, z);
    FPType west =   t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x - 1.0f, y,        z);
    FPType center = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y,        z);
    FPType east =   t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x + 1.0f, y,        z);
    FPType north =  t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y + 1.0f, z);
    FPType far =    t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y,        z + 1.0f);
    FPType b =      t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), x,        y,        z);

    handler.HandleUpperBoundary(&north, center, y, volume_size.y);

    FPType beta = 6.0f;
    ModifyBoundaryCoef(&beta, x, y, z, volume_size);

    float u = (-omega * center + center) +
        (west + east + south + north + far + near - b) * omega / beta;

    t3d.Store(u, surf, x, y, z);
}

__device__ void ReadBlockAndHalo_32x6(int z, uint tx, uint ty, float2* smem)
{
    uint linear_index = ty * blockDim.x + tx;

    const uint smem_width = 48;

    uint sx =  linear_index % smem_width;
    uint sy1 = linear_index / smem_width;
    uint sy2 = sy1 + 4;

    int ix =  static_cast<int>(blockIdx.x * blockDim.x + sx) - 8;
    int iy1 = static_cast<int>(blockIdx.y * blockDim.y + sy1) - 1;
    int iy2 = static_cast<int>(blockIdx.y * blockDim.y + sy2) - 1;

    smem[sx + sy1 * smem_width] = tex3D(tex_packed, ix, iy1, z);
    smem[sx + sy2 * smem_width] = tex3D(tex_packed, ix, iy2, z);
}

__device__ void SaveToRegisters(float2* smem, uint si, uint bw, float* south,
                                float* west, float2* center, float* east,
                                float* north)
{
    __syncthreads();

    *south =  smem[si - bw].x;
    *west =   smem[si - 1].x;
    *center = smem[si];
    *east =   smem[si + 1].x;
    *north =  smem[si + bw].x;
}

__global__ void DampedJacobiKernel_smem_25d_32x6(float omega_over_beta,
                                                 int z0, int z2, int zi, int zn,
                                                 uint3 volume_size)
{
    __shared__ float2 smem[384];

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint bw = blockDim.x + 16;
    const uint ox = blockIdx.x * blockDim.x + tx;
    const uint oy = blockIdx.y * blockDim.y + ty;

    const uint si = (ty + 1) * bw + tx + 8;

    float  south;
    float  west;
    float2 center;
    float  east;
    float  north;

    ReadBlockAndHalo_32x6(z0, tx, ty, smem);
    SaveToRegisters(smem, si, bw, &south, &west, &center, &east, &north);

    ReadBlockAndHalo_32x6(z0 + zi, tx, ty, smem);

    float t1 = omega_over_beta * 4.0f * center.x +
        (west + east + south + north - center.y) * omega_over_beta;
    float b = center.y;
    float near = t1;

    ushort2 raw;
    float far;

    // If we replace the initial value of |z| to "z0 + zi + zi", the compiler
    // of CUDA 7.5 will generate some weird code that affects the mechanism of
    // updating the memory in-place, which slows down the speed of converge
    // a lot. A new variable "z2" is added to be a workaround.
    for (int z = z2; z != zn; z += zi) {
        SaveToRegisters(smem, si, bw, &south, &west, &center, &east, &north);
        ReadBlockAndHalo_32x6(z, tx, ty, smem);

        far = center.x * omega_over_beta;
        near = t1 + far;
        raw = make_ushort2(__float2half_rn(near), __float2half_rn(b));
        if (oy < volume_size.y)
            surf3Dwrite(raw, surf, ox * sizeof(ushort2), oy, z - zi - zi,
            cudaBoundaryModeTrap);

        // t1 is now pointing to plane |i - 1|.
        t1 = omega_over_beta * 3.0f * center.x +
            (west + east + south + north + near - center.y) * omega_over_beta;
        b = center.y;
    }

    SaveToRegisters(smem, si, bw, &south, &west, &center, &east, &north);
    if (oy >= volume_size.y)
        return;

    near = center.x * omega_over_beta + t1;
    raw = make_ushort2(__float2half_rn(near),
                       __float2half_rn(b));
    surf3Dwrite(raw, surf, ox * sizeof(ushort2), oy, zn - zi - zi,
                cudaBoundaryModeTrap);

    t1 = omega_over_beta * 4.0f * center.x +
        (west + east + south + north + near - center.y) * omega_over_beta;
    raw = make_ushort2(__float2half_rn(t1), __float2half_rn(center.y));
    surf3Dwrite(raw, surf, ox * sizeof(ushort2), oy, zn - zi,
                cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_branch(float omega_over_beta,
                                               uint3 volume_size)
{
    __shared__ float2 cached_block[1000];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int bw = blockDim.x + 2;
    int bh = blockDim.y + 2;

    int index = (threadIdx.z + 1) * bw * bh + (threadIdx.y + 1) * bw +
        threadIdx.x + 1;
    cached_block[index] = tex3D(tex_packed, x, y, z);

    if (threadIdx.x == 0)
        cached_block[index - 1] = x == 0 ?                       cached_block[index] : tex3D(tex_packed, x - 1, y, z);

    if (threadIdx.x == blockDim.x - 1)
        cached_block[index + 1] = x == volume_size.x - 1 ?       cached_block[index] : tex3D(tex_packed, x + 1, y, z);

    if (threadIdx.y == 0)
        cached_block[index - bw] = y == 0 ?                      cached_block[index] : tex3D(tex_packed, x, y - 1, z);

    if (threadIdx.y == blockDim.y - 1)
        cached_block[index + bw] = y == volume_size.y - 1 ?      cached_block[index] : tex3D(tex_packed, x, y + 1, z);

    if (threadIdx.z == 0)
        cached_block[index - bw * bh] = z == 0 ?                 cached_block[index] : tex3D(tex_packed, x, y, z - 1);

    if (threadIdx.z == blockDim.z - 1)
        cached_block[index + bw * bh] = z == volume_size.z - 1 ? cached_block[index] : tex3D(tex_packed, x, y, z + 1);

    __syncthreads();

    float  near =   cached_block[index - bw * bh].x;
    float  south =  cached_block[index - bw].x;
    float  west =   cached_block[index - 1].x;
    float2 center = cached_block[index];
    float  east =   cached_block[index + 1].x;
    float  north =  cached_block[index + bw].x;
    float  far =    cached_block[index + bw * bh].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near - center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_assist_thread(float omega_over_beta)
{
    // Shared memory solution with halo handled by assistant threads still
    // runs a bit slower than the texture-only way(less than 3ms on my GTX
    // 660Ti doing 40 times Jacobi).
    //
    // With the bank conflicts solved, I think the difference can be narrowed
    // down to around 1ms. But, it may say that the power of shared memory is
    // not as that great as expected, for Jacobi at least. Or maybe the texture
    // cache is truely really fast.

    const int cache_size = 1000;
    const int bd = 10;
    const int bh = 10;
    const int slice_stride = cache_size / bd;
    const int bw = slice_stride / bh;

    __shared__ float2 cached_block[cache_size];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = (threadIdx.z + 1) * slice_stride + (threadIdx.y + 1) * bw +
        threadIdx.x + 1;

    // Kernel runs faster if we place the normal fetch prior to the assistant
    // process.
    cached_block[index] = tex3D(tex_packed, x, y, z);

    int inner = 0;
    int inner_x = 0;
    int inner_y = 0;
    int inner_z = 0;
    switch (threadIdx.z) {
        case 0: {
            // near
            inner = (threadIdx.y + 1) * bw + threadIdx.x + 1;
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z - 1;
            cached_block[inner] = tex3D(tex_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 1: {
            // south
            inner = (threadIdx.y + 1) * slice_stride + threadIdx.x + 1;
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y - 1;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_block[inner] = tex3D(tex_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 2: {
            // west
            inner = (threadIdx.x + 1) * slice_stride + (threadIdx.y + 1) * bw;

            // It's more efficient to put z in the inner-loop than y.
            inner_x = blockIdx.x * blockDim.x - 1;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_block[inner] = tex3D(tex_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 5:
            // east
            inner = (threadIdx.x + 1) * slice_stride + (threadIdx.y + 1) * bw +
                blockDim.x + 1;
            inner_x = blockIdx.x * blockDim.x + blockDim.x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_block[inner] = tex3D(tex_packed, inner_x, inner_y,
                                        inner_z);
            break;
        case 6:
            // north
            inner = (threadIdx.y + 1) * slice_stride + (blockDim.y + 1) * bw +
                threadIdx.x + 1;
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y + blockDim.y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_block[inner] = tex3D(tex_packed, inner_x, inner_y,
                                        inner_z);
            break;
        case 7:
            // far
            inner = (blockDim.z + 1) * slice_stride + (threadIdx.y + 1) * bw +
                threadIdx.x + 1;
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + blockDim.z;
            cached_block[inner] = tex3D(tex_packed, inner_x, inner_y,
                                        inner_z);
            break;
    }
    __syncthreads();

    float  near =   cached_block[index - slice_stride].x;
    float  south =  cached_block[index - bw].x;
    float  west =   cached_block[index - 1].x;
    float2 center = cached_block[index];
    float  east =   cached_block[index + 1].x;
    float  north =  cached_block[index + bw].x;
    float  far =    cached_block[index + slice_stride].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near - center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_faces_assist_thread(
    float omega_over_beta)
{
    const int cache_size = 512;
    const int bd = 8;
    const int bh = 8;
    const int slice_stride = cache_size / bd;
    const int bw = slice_stride / bh;

    __shared__ float2 cached_block[cache_size];
    __shared__ float cached_face_xyz0[bw * bh];
    __shared__ float cached_face_xyz1[bw * bh];
    __shared__ float cached_face_xzy0[bw * bd];
    __shared__ float cached_face_xzy1[bw * bd];
    __shared__ float cached_face_yzx0[bh * bd];
    __shared__ float cached_face_yzx1[bh * bd];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = threadIdx.z * slice_stride + threadIdx.y * bw + threadIdx.x;

    cached_block[index] = tex3D(tex_packed, x, y, z);

    int inner_x = 0;
    int inner_y = 0;
    int inner_z = 0;
    switch (threadIdx.z) {
        case 0: {
            // near
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z - 1;
            cached_face_xyz0[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(tex_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 1: {
            // south
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y - 1;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_face_xzy0[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(tex_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 2: {
            // west
            inner_x = blockIdx.x * blockDim.x - 1;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_face_yzx0[blockDim.y * threadIdx.y + threadIdx.x] =
                tex3D(tex_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 5:
            // east
            inner_x = blockIdx.x * blockDim.x + blockDim.x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_face_yzx1[blockDim.y * threadIdx.y + threadIdx.x] =
                tex3D(tex_packed, inner_x, inner_y, inner_z).x;
            break;
        case 6:
            // north
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y + blockDim.y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_face_xzy1[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(tex_packed, inner_x, inner_y, inner_z).x;
            break;
        case 7:
            // far
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + blockDim.z;
            cached_face_xyz1[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(tex_packed, inner_x, inner_y, inner_z).x;
            break;
    }
    __syncthreads();

    float2 center = cached_block[index];
    float near =  threadIdx.z == 0 ?              cached_face_xyz0[blockDim.x * threadIdx.y + threadIdx.x] : cached_block[index - slice_stride].x;
    float south = threadIdx.y == 0 ?              cached_face_xzy0[blockDim.x * threadIdx.z + threadIdx.x] : cached_block[index - bw].x;
    float west =  threadIdx.x == 0 ?              cached_face_yzx0[blockDim.y * threadIdx.y + threadIdx.z] : cached_block[index - 1].x;
    float east =  threadIdx.x == blockDim.x - 1 ? cached_face_yzx1[blockDim.y * threadIdx.y + threadIdx.z] : cached_block[index + 1].x;
    float north = threadIdx.y == blockDim.y - 1 ? cached_face_xzy1[blockDim.x * threadIdx.z + threadIdx.x] : cached_block[index + bw].x;
    float far =   threadIdx.z == blockDim.z - 1 ? cached_face_xyz1[blockDim.x * threadIdx.y + threadIdx.x] : cached_block[index + slice_stride].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near - center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_dedicated_assist_thread(
    float omega_over_beta)
{
    __shared__ float2 cached_block[1000];

    int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
    int z = blockIdx.z * (blockDim.z - 2) + threadIdx.z - 1;

    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;

    cached_block[index] = tex3D(tex_packed, x, y, z);

    __syncthreads();

    if (threadIdx.x < 1 || threadIdx.x > blockDim.x - 2 ||
            threadIdx.y < 1 || threadIdx.y > blockDim.y - 2 ||
            threadIdx.z < 1 || threadIdx.z > blockDim.z - 2)
        return;

    float2 center = cached_block[index];
    float near =    cached_block[index - blockDim.x * blockDim.y].x;
    float south =   cached_block[index - blockDim.x].x;
    float west =    cached_block[index - 1].x;
    float east =    cached_block[index + 1].x;
    float north =   cached_block[index + blockDim.x].x;
    float far =     cached_block[index + blockDim.x * blockDim.y].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near - center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_no_halo_storage(float omega_over_beta,
                                                        uint3 volume_size)
{
    __shared__ float2 cached_block[512];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;

    cached_block[index] = tex3D(tex_packed, x, y, z);
    __syncthreads();

    float center = cached_block[index].x;
    float near =  threadIdx.z == 0 ?              (z == 0 ?                 center : tex3D(tex_packed, x, y, z - 1.0f).x) : cached_block[index - blockDim.x * blockDim.y].x;
    float south = threadIdx.y == 0 ?              (y == 0 ?                 center : tex3D(tex_packed, x, y - 1.0f, z).x) : cached_block[index - blockDim.x].x;
    float west =  threadIdx.x == 0 ?              (x == 0 ?                 center : tex3D(tex_packed, x - 1.0f, y, z).x) : cached_block[index - 1].x;
    float east =  threadIdx.x == blockDim.x - 1 ? (x == volume_size.x - 1 ? center : tex3D(tex_packed, x + 1.0f, y, z).x) : cached_block[index + 1].x;
    float north = threadIdx.y == blockDim.y - 1 ? (y == volume_size.y - 1 ? center : tex3D(tex_packed, x, y + 1.0f, z).x) : cached_block[index + blockDim.x].x;
    float far =   threadIdx.z == blockDim.z - 1 ? (z == volume_size.z - 1 ? center : tex3D(tex_packed, x, y, z + 1.0f).x) : cached_block[index + blockDim.x * blockDim.y].x;

    float b_center = cached_block[index].y;

    float u = omega_over_beta * 3.0f * center +
        (west + east + south + north + far + near - b_center) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(b_center));

    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

template <typename StorageType, typename UpperBoundaryHandler>
__global__ void RelaxRedBlackGaussSeidelKernel(uint3 volume_size, uint offset,
                                               UpperBoundaryHandler handler)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    x = (x << 1) + ((offset + y + z) & 0x1);

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    Tex3d<StorageType> t3d;
    FPType near   = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y,        z - 1.0f);
    FPType south  = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y - 1.0f, z);
    FPType west   = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x - 1.0f, y,        z);
    FPType center = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y,        z);
    FPType east   = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x + 1.0f, y,        z);
    FPType north  = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y + 1.0f, z);
    FPType far    = t3d(TexSel<StorageType>::Tex(tex_u, texf_u, texd_u), x,        y,        z + 1.0f);
    FPType b      = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), x,        y,        z);

    handler.HandleUpperBoundary(&north, center, y, volume_size.y);

    FPType beta = 6.0f;
    ModifyBoundaryCoef(&beta, x, y, z, volume_size);

    // NOTE: The coefficient 'h^2' is premultiplied in the divergence kernel.
    FPType u = (west + east + south + north + far + near - b) / beta;

    t3d.Store(u, surf, x, y, z);
}

template <typename StorageType>
__global__ void RelaxWithZeroGuessKernel(float omega, float coef,
                                         float omega_over_beta,
                                         uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z);

    Tex3d<StorageType> t3d;
    FPType near   = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x,        coord.y,        coord.z - 1.0f);
    FPType south  = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x,        coord.y - 1.0f, coord.z);
    FPType west   = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x - 1.0f, coord.y,        coord.z);
    FPType center = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x,        coord.y,        coord.z);
    FPType east   = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x + 1.0f, coord.y,        coord.z);
    FPType north  = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x,        coord.y + 1.0f, coord.z);
    FPType far    = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), coord.x,        coord.y,        coord.z + 1.0f);

    FPType beta = 6.0f;
    ModifyBoundaryCoef(&beta, x, y, z, volume_size);

    // u0 = omega * -b * 1/6 (ignore boundary condition)
    //    = -1/9 * b
    //
    // u1 = (1 - omega) * u0 + (|u_adj| - b) * omega / beta
    //    = -1 / 27 * b + (-1/9 * |b_adj| - b) * omega / beta

    FPType v = coef * center;
    FPType w = -omega_over_beta * (north + south + east + west + far + near);
    FPType u = (w - center) * omega / beta + v;

    t3d.Store(u, surf, x, y, z);
}

// =============================================================================

template <typename StorageType>
struct RelaxRedBlackGaussSeidelKernelMeta
{
    static void Invoke(const dim3& grid, const dim3& block,
                       const uint3& volume_size, uint offset, bool outflow)
    {
        using FPType = typename Tex3d<StorageType>::ValType;
        UpperBoundaryHandlerOutflow<FPType> outflow_handler;
        UpperBoundaryHandlerNeumann<FPType> neumann_handler;
        if (outflow)
            RelaxRedBlackGaussSeidelKernel<StorageType><<<grid, block>>>(
                volume_size, offset, outflow_handler);
        else
            RelaxRedBlackGaussSeidelKernel<StorageType><<<grid, block>>>(
                volume_size, offset, neumann_handler);
    }
};

template <typename StorageType>
struct DampedJacobiKernelMeta
{
    static void Invoke(const dim3& grid, const dim3& block, float omega,
                       const uint3& volume_size, bool outflow)
    {
        using FPType = typename Tex3d<StorageType>::ValType;
        UpperBoundaryHandlerOutflow<FPType> outflow_handler;
        UpperBoundaryHandlerNeumann<FPType> neumann_handler;
        if (outflow)
            DampedJacobiKernel<StorageType><<<grid, block>>>(
                omega, volume_size, outflow_handler);
        else
            DampedJacobiKernel<StorageType><<<grid, block>>>(
                omega, volume_size, neumann_handler);
    }
};

DECLARE_KERNEL_META(
    RelaxWithZeroGuessKernel,
    MAKE_INVOKE_DECLARATION(float omega, float coef, float omega_over_beta,
                            const uint3& volume_size),
    omega, coef, omega_over_beta, volume_size);

// =============================================================================

void RelaxDampedJacobi(cudaArray* unp1, cudaArray* un, cudaArray* b,
                       bool outflow, int num_of_iterations, uint3 volume_size,
                       BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, unp1) != cudaSuccess)
        return;

    auto bound_u = SelectiveBind(un, false, cudaFilterModePoint,
                                 cudaAddressModeBorder, &tex_u, &texf_u,
                                 &texd_u);
    if (!bound_u.Succeeded())
        return;

    auto bound_b = SelectiveBind(b, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_b, &texf_b,
                                 &texd_b);
    if (!bound_b.Succeeded())
        return;

    float omega_over_beta = 0.11111111f;
    for (int i = 0; i < num_of_iterations; i++) {
        bool smem = false;
        bool smem_25d = false;
        if (smem_25d) {
            // In our tests, this kernel sometimes generated results that
            // noticably worse than the others. It probably connects to the
            // undetermined behavior of updating memory while reading.
            // But we haven't encountered any "obviously-worse" case in the
            // non-shared-memory version, so far, in our experiments.
            dim3 block(32, 6, 1);
            dim3 grid((volume_size.x + block.x - 1) / block.x,
                      (volume_size.y + block.y - 1) / block.y,
                      1);

            if (i >= num_of_iterations / 2)
                DampedJacobiKernel_smem_25d_32x6<<<grid, block>>>(
                    omega_over_beta, volume_size.z - 1, volume_size.z - 3, -1,
                    -1, volume_size);
            else
                DampedJacobiKernel_smem_25d_32x6<<<grid, block>>>(
                    omega_over_beta, 0, 2, 1, volume_size.z, volume_size);
            
        } else if (smem) {
            dim3 block;
            dim3 grid;
            ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
            DampedJacobiKernel_smem_assist_thread<<<grid, block>>>(
                omega_over_beta);
        } else {
            float omega = 2.0f / 3.0f;
            dim3 block;
            dim3 grid;
            ba->ArrangeRowScan(&block, &grid, volume_size);
            InvokeKernel<DampedJacobiKernelMeta>(bound_u, grid, block, omega,
                                                 volume_size, outflow);
        }
    }

    DCHECK_KERNEL();
}

void RelaxRedBlackGaussSeidel(cudaArray* unp1, cudaArray* un, cudaArray* b,
                              bool outflow, int num_of_iterations,
                              uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, unp1) != cudaSuccess)
        return;

    auto bound_u = SelectiveBind(un, false, cudaFilterModePoint,
                                 cudaAddressModeBorder, &tex_u, &texf_u,
                                 &texd_u);
    if (!bound_u.Succeeded())
        return;

    auto bound_b = SelectiveBind(b, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_b, &texf_b,
                                 &texd_b);
    if (!bound_b.Succeeded())
        return;

    uint3 half_size = volume_size;
    half_size.x /= 2;
    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, half_size);

    for (int i = 0; i < num_of_iterations; i++) {
        InvokeKernel<RelaxRedBlackGaussSeidelKernelMeta>(bound_u,  grid, block,
                                                         volume_size, 0,
                                                         outflow);
        InvokeKernel<RelaxRedBlackGaussSeidelKernelMeta>(bound_u, grid, block,
                                                         volume_size, 1,
                                                         outflow);
    }
    DCHECK_KERNEL();
}

namespace kern_launcher
{
void Relax(cudaArray* unp1, cudaArray* un, cudaArray* b, bool outflow,
           int num_of_iterations, uint3 volume_size, BlockArrangement* ba)
{
    bool jacobi = false;;
    if (jacobi)
        RelaxDampedJacobi(unp1, un, b, outflow, num_of_iterations, volume_size,
                          ba);
    else
        RelaxRedBlackGaussSeidel(unp1, un, b, outflow, num_of_iterations,
                                 volume_size, ba);
}

void RelaxWithZeroGuess(cudaArray* u, cudaArray* b, uint3 volume_size,
                        BlockArrangement* ba)
{
    float omega           = 2.0f / 3.0f;
    float omega_over_beta = omega / kBeta;
    float coef            = omega * (omega - 1.0f) / kBeta;

    if (BindCudaSurfaceToArray(&surf, u) != cudaSuccess)
        return;

    auto bound = SelectiveBind(b, false, cudaFilterModePoint,
                               cudaAddressModeBorder, &tex_b, &texf_b, &texd_b);
    if (!bound.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);

    InvokeKernel<RelaxWithZeroGuessKernelMeta>(bound, grid, block, omega, coef,
                                               omega_over_beta, volume_size);
    DCHECK_KERNEL();
}
}
