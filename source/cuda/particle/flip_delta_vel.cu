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

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_xp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_yp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_zp;

namespace
{
template <typename TexType>
__device__ void LoadVelToSmem_10x10x10(float3* smem, TexType tex_x_obj,
                                       TexType tex_y_obj, TexType tex_z_obj,
                                       const uint3& volume_size)
{
    uint smem_block_width = 10;
    uint smem_slice_size = 10 * 10;
    uint smem_stride = 8 * 8 * 8;

    // Linear index in the kernel block.
    uint i = LinearIndexBlock();

    // Shared memory block(including halo) coordinates.
    uint sz = i / smem_slice_size;
    uint t =  i - __umul24(sz, smem_slice_size);
    uint sy = t / smem_block_width;
    uint sx = t - __umul24(sy, smem_block_width);

    uint i2 = i + smem_stride;
    uint sz2 = i2 / smem_slice_size;
    t =  i2 - __umul24(sz2, smem_slice_size);
    uint sy2 = t / smem_block_width;
    uint sx2 = t - __umul24(sy2, smem_block_width);

    // Grid coordinates.
    float gx = __uint2float_rn(blockIdx.x * blockDim.x + sx - 1) + 0.5f;
    float gy = __uint2float_rn(blockIdx.y * blockDim.y + sy - 1) + 0.5f;
    float gz = __uint2float_rn(blockIdx.z * blockDim.z + sz - 1) + 0.5f;

    float gx2 = __uint2float_rn(blockIdx.x * blockDim.x + sx2 - 1) + 0.5f;
    float gy2 = __uint2float_rn(blockIdx.y * blockDim.y + sy2 - 1) + 0.5f;
    float gz2 = __uint2float_rn(blockIdx.z * blockDim.z + sz2 - 1) + 0.5f;

    smem[i              ].x = tex3D(tex_x_obj, gx  + 0.5f, gy,         gz);
    smem[i + smem_stride].x = tex3D(tex_x_obj, gx2 + 0.5f, gy2,        gz2);

    smem[i              ].y = tex3D(tex_y_obj, gx,         gy  + 0.5f, gz);
    smem[i + smem_stride].y = tex3D(tex_y_obj, gx2,        gy2 + 0.5f, gz2);

    smem[i              ].z = tex3D(tex_z_obj, gx,         gy ,        gz  + 0.5f);
    smem[i + smem_stride].z = tex3D(tex_z_obj, gx2,        gy2,        gz2 + 0.5f);
}

__device__ float3 LoadVelFromSmem_10x10xX(const float3* smem,
                                          const float3& offset, uint si)
{
    int3 p0 = make_int3(__float2int_rd(offset.x), __float2int_rd(offset.y),
                        __float2int_rd(offset.z));
    float3 t = offset - make_float3(__int2float_rn(p0.x), __int2float_rn(p0.y),
                                    __int2float_rn(p0.z));

    // p0 should always stay between [-1, 1].

    int row_stride = 10;
    int slice_stride = 10 * 10;

    int i = si + p0.z * slice_stride + p0.y * row_stride + p0.x;

    float3 x0y0z0 = smem[i];
    float3 x1y0z0 = smem[i + 1];
    float3 x0y1z0 = smem[i + row_stride];
    float3 x0y0z1 = smem[i + slice_stride];
    float3 x1y1z0 = smem[i + row_stride + 1];
    float3 x0y1z1 = smem[i + slice_stride + row_stride];
    float3 x1y0z1 = smem[i + slice_stride + 1];
    float3 x1y1z1 = smem[i + slice_stride + row_stride + 1];

    float3 a = lerp(x0y0z0, x1y0z0, t.x);
    float3 b = lerp(x0y1z0, x1y1z0, t.x);
    float3 c = lerp(x0y0z1, x1y0z1, t.x);
    float3 d = lerp(x0y1z1, x1y1z1, t.x);

    float3 e = lerp(a, b, t.y);
    float3 f = lerp(c, d, t.y);

    return lerp(e, f, t.z);
}

// Both attempts using shared memory to accelerate this interpolation had
// failed. The major problem is the number of registers.
//
// To raise the rate of re-using shared memory, we need to configure the block
// as large as we can, or the halo part would become relatively expensive(
// redundant across blocks, more reads from texture for each thread). However,
// large blocks means fewer registers per thread. In the 8x8x8 case, each
// thread would have a cap of registers count of 64K / (512 * 4) = 32, which is
// definitely not enough for a tri-linear interpolation kernel.
//
// Besides, compared to the normal version, shared memory version introduces
// more execution divergence, and loses the hardware-accelerated tri-linear
// interpolation, which is also not negligible to performance.
//
// Moreover, this FLIP system now has a flaw, that due to the fp16
// presentation of the position fields, some particles generated from one cell
// may be 'moved' to another when the rounding happens. After the emission
// kernel, there are always some particles near the cell faces get drifted, and
// this breaks the assumption that every particle should be no further than
// 1 cell to the current cell center. And there is not yet a trivial way fixing
// this problem.
//
// So, this is it. I consider it as a dead-end.
__global__ void InterpolateDeltaVelocityKernel_smem(
    uint16_t* vel_x, uint16_t* vel_y, uint16_t* vel_z,
    const uint16_t* __restrict__ pos_x, const uint16_t* __restrict__ pos_y,
    const uint16_t* __restrict__ pos_z, uint3 volume_size)
{
    __shared__ float3 smem[1024];
    __shared__ float3 smemp[1024];

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    bool inside_volume =
        !(x >= volume_size.x || y >= volume_size.y || z >= volume_size.z);

    float3 center = make_float3(x, y, z) + 0.5f;

    // Coordinates in the shared memory block(including halo).
    uint3 ti = threadIdx + 1;

    // Linear index in the shared memory block(including halo).
    uint si = __umul24(__umul24(ti.z, 10) + ti.y, 10) + ti.x;

    uint cell_index = LinearIndexVolume(x, y, z, volume_size);
    uint index = ParticleIndex(cell_index);

    LoadVelToSmem_10x10x10(smem,  tex_x,  tex_y,  tex_z,  volume_size);
    LoadVelToSmem_10x10x10(smemp, tex_xp, tex_yp, tex_zp, volume_size);
    __syncthreads();

    for (int i = 0; inside_volume && i < kMaxNumParticlesPerCell; i++) {
        if (IsParticleUndefined(pos_x[index + i]))
            break; // All active particles within a cell should be consecutive.
        
        float3 coord = flip::Position32(pos_x[index + i], pos_y[index + i],
                                        pos_z[index + i]);

        float3 vel  = LoadVelFromSmem_10x10xX(smem,  coord - center, si);
        float3 velp = LoadVelFromSmem_10x10xX(smemp, coord - center, si);
        float3 delta = velp - vel;

        // v_np1 = (1 - ¦Á) * v_n_pic + ¦Á * v_n_flip.
        // We are using ¦Á = 1.
        vel_x[index + i] = __float2half_rn(__half2float(vel_x[index + i]) + delta.x);
        vel_y[index + i] = __float2half_rn(__half2float(vel_y[index + i]) + delta.y);
        vel_z[index + i] = __float2half_rn(__half2float(vel_z[index + i]) + delta.z);
    }
}

template <typename TexType>
__device__ void LoadVelToSmemNoHalo(float3* smem, TexType tex_x_obj,
                                    TexType tex_y_obj, TexType tex_z_obj,
                                    const uint3& volume_size)
{
    // Linear index in the kernel block.
    uint i = LinearIndexBlock();

    // Grid coordinates.
    float gx = __uint2float_rn(blockIdx.x * blockDim.x + threadIdx.x) + 0.5f;
    float gy = __uint2float_rn(blockIdx.y * blockDim.y + threadIdx.y) + 0.5f;
    float gz = __uint2float_rn(blockIdx.z * blockDim.z + threadIdx.z) + 0.5f;

    smem[i].x = tex3D(tex_x_obj, gx + 0.5f, gy,        gz);
    smem[i].y = tex3D(tex_y_obj, gx,        gy + 0.5f, gz);
    smem[i].z = tex3D(tex_z_obj, gx,        gy ,       gz + 0.5f);
}

template <int bw, int bh, int bd, typename TexType>
__device__ float3 LoadVelFromSmemNoHalo(const float3* smem, TexType tex_x_obj,
                                        TexType tex_y_obj, TexType tex_z_obj,
                                        const float3& coord,
                                        const float3& center, uint si)
{
    float3 offset = coord - center;
    int3 p0 = make_int3(__float2int_rd(offset.x), __float2int_rd(offset.y),
                        __float2int_rd(offset.z));
    float3 t = offset - make_float3(__int2float_rn(p0.x), __int2float_rn(p0.y),
                                    __int2float_rn(p0.z));

    // p0 should always stay between [-1, 1].

    int row_stride = bw;
    int slice_stride = bd * bh;

    int i = si + p0.z * slice_stride + p0.y * row_stride + p0.x;
    if (i < 0 || i > (slice_stride * bd - slice_stride - row_stride - 2)) {
        return make_float3(
            tex3D(tex_x_obj, coord.x + 0.5f, coord.y,        coord.z),
            tex3D(tex_y_obj, coord.x,        coord.y + 0.5f, coord.z),
            tex3D(tex_z_obj, coord.x,        coord.y ,       coord.z + 0.5f));
    }

    float3 x0y0z0 = smem[i];
    float3 x1y0z0 = smem[i + 1];
    float3 x0y1z0 = smem[i + row_stride];
    float3 x0y0z1 = smem[i + slice_stride];
    float3 x1y1z0 = smem[i + row_stride + 1];
    float3 x0y1z1 = smem[i + slice_stride + row_stride];
    float3 x1y0z1 = smem[i + slice_stride + 1];
    float3 x1y1z1 = smem[i + slice_stride + row_stride + 1];

    float3 a = lerp(x0y0z0, x1y0z0, t.x);
    float3 b = lerp(x0y1z0, x1y1z0, t.x);
    float3 c = lerp(x0y0z1, x1y0z1, t.x);
    float3 d = lerp(x0y1z1, x1y1z1, t.x);

    float3 e = lerp(a, b, t.y);
    float3 f = lerp(c, d, t.y);

    return lerp(e, f, t.z);
}

__global__ void InterpolateDeltaVelocityKernel_smem_8x8x8_no_halo(
    uint16_t* vel_x, uint16_t* vel_y, uint16_t* vel_z, const uint16_t* pos_x,
    const uint16_t* pos_y, const uint16_t* pos_z,
    const uint32_t* particle_count, uint3 volume_size)
{
    __shared__ float3 smem[512];
    __shared__ float3 smemp[512];

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    bool inside_volume =
        !(x >= volume_size.x || y >= volume_size.y || z >= volume_size.z);

    float3 center = make_float3(x, y, z) + 0.5f;

    // Linear index in the shared memory block(including halo).
    uint si = LinearIndexBlock();

    uint cell_index = LinearIndexVolume(x, y, z, volume_size);
    uint index = ParticleIndex(cell_index);
    int count = inside_volume ? particle_count[cell_index] : 0;
    if (count) { // Little help to performance, since most of the cells are
                 // not empty.
        LoadVelToSmemNoHalo(smem,  tex_x,  tex_y,  tex_z,  volume_size);
        LoadVelToSmemNoHalo(smemp, tex_xp, tex_yp, tex_zp, volume_size);
    }

    __syncthreads();

    for (int i = 0; i < count; i++) {
        // IsCellUndefined(pos_x[index + i])) should always return *TRUE*.
        // All active particles within a cell should be consecutive.
        float3 coord = flip::Position32(pos_x[index + i], pos_y[index + i],
                                        pos_z[index + i]);

        float3 vel  = LoadVelFromSmemNoHalo<8,8,8>(smem,   tex_x,  tex_y,
                                                   tex_z,  coord,  center, si);
        float3 velp = LoadVelFromSmemNoHalo<8,8,8>(smemp,  tex_xp, tex_yp,
                                                   tex_zp, coord,  center, si);
        float3 delta = velp - vel;

        // v_np1 = (1 - ¦Á) * v_n_pic + ¦Á * v_n_flip.
        // We are using ¦Á = 1.
        vel_x[index + i] = __float2half_rn(__half2float(vel_x[index + i]) + delta.x);
        vel_y[index + i] = __float2half_rn(__half2float(vel_y[index + i]) + delta.y);
        vel_z[index + i] = __float2half_rn(__half2float(vel_z[index + i]) + delta.z);
    }
}

// Should be invoked *BEFORE* resample kernel. Please read the comments of
// ResampleKernel().
// Active particles are always *NOT* consecutive.
__global__ void InterpolateDeltaVelocityKernel(uint16_t* vel_x, uint16_t* vel_y,
                                               uint16_t* vel_z,
                                               const uint16_t* pos_x,
                                               const uint16_t* pos_y,
                                               const uint16_t* pos_z,
                                               int num_of_particles)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num_of_particles)
        return;

    if (IsParticleUndefined(pos_x[i]))
        return;

    float3 pos = flip::Position32(pos_x[i], pos_y[i], pos_z[i]);

    float v_x =  tex3D(tex_x,  pos.x + 0.5f, pos.y,        pos.z);
    float v_y =  tex3D(tex_y,  pos.x,        pos.y + 0.5f, pos.z);
    float v_z =  tex3D(tex_z,  pos.x,        pos.y,        pos.z + 0.5f);

    float v_xp = tex3D(tex_xp, pos.x + 0.5f, pos.y,        pos.z);
    float v_yp = tex3D(tex_yp, pos.x,        pos.y + 0.5f, pos.z);
    float v_zp = tex3D(tex_zp, pos.x,        pos.y,        pos.z + 0.5f);

    float ¦Ä_x = v_xp - v_x;
    float ¦Ä_y = v_yp - v_y;
    float ¦Ä_z = v_zp - v_z;

    // v_np1 = (1 - ¦Á) * v_n_pic + ¦Á * v_n_flip.
    // We are using ¦Á = 1.
    vel_x[i] = __float2half_rn(__half2float(vel_x[i]) + ¦Ä_x);
    vel_y[i] = __float2half_rn(__half2float(vel_y[i]) + ¦Ä_y);
    vel_z[i] = __float2half_rn(__half2float(vel_z[i]) + ¦Ä_z);
}
} // Anonymous namespace.

// =============================================================================

namespace kern_launcher
{
void InterpolateDeltaVelocity(const FlipParticles& particles, cudaArray* vnp1_x,
                              cudaArray* vnp1_y, cudaArray* vnp1_z,
                              cudaArray* vn_x, cudaArray* vn_y, cudaArray* vn_z,
                              const uint3& volume_size, BlockArrangement* ba)
{
    auto bound_xp = BindHelper::Bind(&tex_xp, vnp1_x, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_xp.error() != cudaSuccess)
        return;

    auto bound_yp = BindHelper::Bind(&tex_yp, vnp1_y, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_yp.error() != cudaSuccess)
        return;

    auto bound_zp = BindHelper::Bind(&tex_zp, vnp1_z, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_zp.error() != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vn_x, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vn_y, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vn_z, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;

    int smem = 0;
    if (smem) {
        if (smem > 1) {
            int bw = 16;
            int bh = 8;
            int bd = 4;
            block = dim3(bw, bh, bd);
            grid = dim3((volume_size.x + bw - 1) / bw,
                        (volume_size.y + bh - 1) / bh,
                        (volume_size.z + bd - 1) / bd);
            InterpolateDeltaVelocityKernel_smem_8x8x8_no_halo<<<grid, block>>>(
                particles.velocity_x_, particles.velocity_y_,
                particles.velocity_z_, particles.position_x_,
                particles.position_y_, particles.position_z_,
                particles.particle_count_, volume_size);
        } else {
            ba->Arrange8x8x8(&grid, &block, volume_size);
            InterpolateDeltaVelocityKernel_smem<<<grid, block>>>(
                particles.velocity_x_, particles.velocity_y_,
                particles.velocity_z_, particles.position_x_,
                particles.position_y_, particles.position_z_,
                volume_size);
        }
    } else {
        ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);
        InterpolateDeltaVelocityKernel<<<grid, block>>>(
            particles.velocity_x_, particles.velocity_y_, particles.velocity_z_,
            particles.position_x_, particles.position_y_, particles.position_z_,
            particles.num_of_particles_);
    }
    DCHECK_KERNEL();
}
}
