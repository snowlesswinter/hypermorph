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
#include <functional>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
surface<void, cudaSurfaceType3D> surf_d;
surface<void, cudaSurfaceType3D> surf_t;

namespace flip
{
//const uint32_t kMaxParticlesInCell = 4;
//const uint16_t kInvalidPos = 48128; // __float2half_rn(-1.0f);

struct ParticleFields
{
    uint16_t* position_x_;
    uint16_t* position_y_;
    uint16_t* position_z_;
    uint16_t* velocity_x_;
    uint16_t* velocity_y_;
    uint16_t* velocity_z_;
    uint16_t* density_;
    uint16_t* temperature_;
};

struct FieldWeightAndWeightedSum
{
    float weight_vel_x_;
    float weight_vel_y_;
    float weight_vel_z_;
    float weight_density_;
    float weight_temperature_;
    float sum_vel_x_;
    float sum_vel_y_;
    float sum_vel_z_;
    float sum_density_;
    float sum_temperature_;
};

struct TemporaryStorage
{
    __device__ inline void Store(float* field, float value)
    {
        *field = value;
    }
};

struct AccumulativeStorage
{
    __device__ inline void Store(float* field, float value)
    {
        *field += value;
    }
};

union FloatSignExtraction
{
    float f_;
    struct
    {
        unsigned int mantisa_ : 23;
        unsigned int exponent_ : 8;
        unsigned int sign_ : 1;
    } parts_;
};

// =============================================================================

__device__ inline float WeightKernel(float p, float p0)
{
    return 1.0f - __saturatef(fabsf(p - p0));
}

__device__ float DistanceWeight(const float3& p, const float3& p0)
{
    return WeightKernel(p.x, p0.x) * WeightKernel(p.y, p0.y) * WeightKernel(p.z, p0.z);
}

template <int Pow>
__global__ void TransferToGridLopKernel(FlipParticles particles,
                                        uint3 volume_size)
{
    const int step = 1 << Pow;

    // save the lookup table into smem
    __shared__ float smem_vx_wsum  [10 * 6 * 6];
    __shared__ float smem_vy_wsum  [10 * 6 * 6];
    __shared__ float smem_vz_wsum  [10 * 6 * 6];
    __shared__ float smem_d_wsum   [10 * 6 * 6];
    __shared__ float smem_t_wsum   [10 * 6 * 6];
    __shared__ float smem_vx_weight[10 * 6 * 6];
    __shared__ float smem_vy_weight[10 * 6 * 6];
    __shared__ float smem_vz_weight[10 * 6 * 6];
    __shared__ float smem_c_weight [10 * 6 * 6];

    int smem_row_stride = 10;
    int smem_slice_stride = 10 * 6;

    uint i = LinearIndex() << Pow;
    for (int p = 0; p < step; p++, i++) {
        if (i >= particles.num_of_particles_) // TODO: Should be constrained
            return;                           //       by kernel configuration.

        // Active particles should be consecutive.
        if (IsParticleUndefined(particles.position_x_[i])) 
            return;

        float3 coord = flip::Position(particles, i);
        float3 corner = floorf(coord);

        // Classify the particle by its situation of adjacency.
        float3 offset = floorf((coord - corner) * 2.0f);
        int3 offset_i = Float2Int(offset);

        float3 center = corner + 0.5f;
        for (int i = -1; i < 1; i++) for (int j = -1; j < 1; j++) for (int k = -1; k < 1; k++) {
            float3 ref_point = center - make_float3(k, j, i) + offset;
            int3 o = offset_i + make_int3(k, j, i);
            float weight = DistanceWeight(coord, ref_point);
            int si = LinearIndexBlock() >> Pow;

            // Careful of the sign of the numbers.
            si += o.z * smem_slice_stride + o.y * smem_row_stride + o.x;

            float3 ref_point_x = center - make_float3(k, j, i) + make_float3(1.0f, offset.y, offset.z);
            float3 ref_point_y = center - make_float3(k, j, i) + make_float3(offset.x, 1.0f, offset.z);
            float3 ref_point_z = center - make_float3(k, j, i) + make_float3(offset.x, offset.y, 1.0f);

            float wx = DistanceWeight(coord, ref_point_x);
            float wy = DistanceWeight(coord, ref_point_y);
            float wz = DistanceWeight(coord, ref_point_z);

            atomicAdd(smem_vx_weight + si, wx);
            atomicAdd(smem_vy_weight + si, wy);
            atomicAdd(smem_vz_weight + si, wz);

            float vx = __half2float(particles.velocity_x_[i]);
            float vy = __half2float(particles.velocity_y_[i]);
            float vz = __half2float(particles.velocity_z_[i]);

            atomicAdd(smem_vx_wsum + si, weight * vx);
            atomicAdd(smem_vy_wsum + si, weight * vy);
            atomicAdd(smem_vz_wsum + si, weight * vz);

            atomicAdd(smem_c_weight + si, weight);

            float density = __half2float(particles.density_[i]);
            atomicAdd(smem_d_wsum + si, weight * density);

            float temperature = __half2float(particles.temperature_[i]);
            atomicAdd(smem_t_wsum + si, weight * temperature);
        }
    }
}
}

namespace kern_launcher
{
void TransferToGridOpt(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                       cudaArray* density, cudaArray* temperature,
                       const FlipParticles& particles, const FlipParticles& aux,
                       uint3 volume_size, BlockArrangement* ba)
{
    
}
}
