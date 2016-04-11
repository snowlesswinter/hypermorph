#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"

surface<void, cudaTextureType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;
surface<void, cudaTextureType3D> buoyancy_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_temperature;
surface<void, cudaTextureType3D> impulse_dest;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> impulse_original;
surface<void, cudaTextureType3D> divergence_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> divergence_velocity;
surface<void, cudaTextureType3D> gradient_velocity;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> gradient_packed;
surface<void, cudaTextureType3D> jacobi;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> jacobi_packed;
surface<void, cudaTextureType3D> diagnosis;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> diagnosis_source;

__global__ void AdvectPureKernel(float time_step, float dissipation,
                                 int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float result = (1.0f - dissipation * time_step) *
        tex3D(advect_source, back_traced.x, back_traced.y, back_traced.z);
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectVelocityPureKernel(float time_step, float dissipation,
                                         int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float4 new_velocity = (1.0f - dissipation * time_step) *
        tex3D(advect_velocity, back_traced.x,
                                              back_traced.y, back_traced.z);
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__device__ float3 f42f3(float4 v)
{
    return make_float3(v.x, v.y, v.z);
}

__device__ float TrilinearInterpolationSingle(float x0y0z0, float x1y0z0,
                                              float x0y1z0, float x0y0z1,
                                              float x1y1z0, float x0y1z1,
                                              float x1y0z1, float x1y1z1,
                                              float ��, float ��, float ��)
{
    float xy0z0 = (1 - ��) * x0y0z0 + �� * x1y0z0;
    float xy1z0 = (1 - ��) * x0y1z0 + �� * x1y1z0;
    float xy0z1 = (1 - ��) * x0y0z1 + �� * x1y0z1;
    float xy1z1 = (1 - ��) * x0y1z1 + �� * x1y1z1;

    float yz0 = (1 - ��) * xy0z0 + �� * xy1z0;
    float yz1 = (1 - ��) * xy0z1 + �� * xy1z1;

    return (1 - ��) * yz0 + �� * yz1;
}

__device__ float3 TrilinearInterpolation(float3* cache, float3 coord,
                                         int slice_stride, int row_stride)
{
    float int_x = floorf(coord.x);
    float int_y = floorf(coord.y);
    float int_z = floorf(coord.z);

    float �� = fracf(coord.x);
    float �� = fracf(coord.y);
    float �� = fracf(coord.z);

    int index = int_z * slice_stride + int_y * row_stride + int_x;
    float3 x0y0z0 = cache[index];
    float3 x1y0z0 = cache[index + 1];
    float3 x0y1z0 = cache[index + row_stride];
    float3 x0y0z1 = cache[index + slice_stride];
    float3 x1y1z0 = cache[index + row_stride + 1];
    float3 x0y1z1 = cache[index + slice_stride + row_stride];
    float3 x1y0z1 = cache[index + slice_stride + 1];
    float3 x1y1z1 = cache[index + slice_stride + row_stride + 1];

    float x = TrilinearInterpolationSingle(x0y0z0.x, x1y0z0.x, x0y1z0.x, x0y0z1.x, x1y1z0.x, x0y1z1.x, x1y0z1.x, x1y1z1.x, ��, ��, ��);
    float y = TrilinearInterpolationSingle(x0y0z0.y, x1y0z0.y, x0y1z0.y, x0y0z1.y, x1y1z0.y, x0y1z1.y, x1y0z1.y, x1y1z1.y, ��, ��, ��);
    float z = TrilinearInterpolationSingle(x0y0z0.z, x1y0z0.z, x0y1z0.z, x0y0z1.z, x1y1z0.z, x0y1z1.z, x1y0z1.z, x1y1z1.z, ��, ��, ��);
    return make_float3(x, y, z);
}

// Only ~45% hit rate, serious block effect, deprecated.
__global__ void AdvectVelocityPureKernel_smem(float time_step,
                                              float dissipation,
                                              int3 volume_size)
{
    __shared__ float3 cached_block[600];

    int base_x = blockIdx.x * blockDim.x;
    int base_y = blockIdx.y * blockDim.y;
    int base_z = blockIdx.z * blockDim.z;

    int x = base_x + threadIdx.x;
    int y = base_y + threadIdx.y;
    int z = base_z + threadIdx.z;

    int bw = blockDim.x;
    int bh = blockDim.y;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    int index = threadIdx.z * bw * bh + threadIdx.y * bw + threadIdx.x;
    cached_block[index] = f42f3(tex3D(advect_velocity, coord.x, coord.y, coord.z));
    float3 velocity = cached_block[index];
    __syncthreads();

    float3 back_traced = coord - time_step * velocity;

    float3 new_velocity;
    if (back_traced.x >= base_x + 0.5f && back_traced.x < base_x + blockDim.x + 0.5f &&
            back_traced.y >= base_y + 0.5f && back_traced.y < base_y + blockDim.y + 0.5f &&
            back_traced.z >= base_z + 0.5f && back_traced.z < base_z + blockDim.z + 0.5f) {

        new_velocity = TrilinearInterpolation(
            cached_block, back_traced - make_float3(base_x + 0.5f, base_y + 0.5f, base_z + 0.5f),
            bw * bh, bw);
    } else {
        new_velocity = f42f3(
            tex3D(advect_velocity, back_traced.x, back_traced.y,
                  back_traced.z));
    }
    new_velocity *= 1.0f - dissipation * time_step;
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ApplyBuoyancyPureKernel(float time_step,
                                        float ambient_temperature,
                                        float accel_factor, float gravity,
                                        int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float4 velocity = tex3D(buoyancy_velocity, coord.x, coord.y, coord.z);
    float t = tex3D(buoyancy_temperature, coord.x, coord.y, coord.z);

    ushort4 result = make_ushort4(__float2half_rn(velocity.x),
                                  __float2half_rn(velocity.y),
                                  __float2half_rn(velocity.z),
                                  0);
    if (t > ambient_temperature) {
        float accel = time_step * ((t - ambient_temperature) * accel_factor -
                                   gravity);
        result.y = __float2half_rn(velocity.y + accel);
    }
    surf3Dwrite(result, buoyancy_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ApplyImpulsePureKernel(float3 center_point, float3 hotspot,
                                       float radius, float value,
                                       int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    if (coord.x > 1.0f && coord.y < 3.0f) {
        float2 diff = make_float2(coord.x, coord.z) -
            make_float2(center_point.x, center_point.z);
        float d = hypotf(diff.x, diff.y);
        if (d < radius) {
            diff = make_float2(coord.x, coord.z) -
                make_float2(hotspot.x, hotspot.z);
            float scale = (radius - hypotf(diff.x, diff.y)) / radius;
            scale = fmaxf(scale, 0.5f);
            surf3Dwrite(__float2half_rn(scale * value), impulse_dest,
                        x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
            return;
        }
    }
}

__global__ void ComputeDivergencePureKernel(float half_inverse_cell_size,
                                            int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float4 near =   tex3D(divergence_velocity, coord.x, coord.y, coord.z - 1.0f);
    float4 south =  tex3D(divergence_velocity, coord.x, coord.y - 1.0f, coord.z);
    float4 west =   tex3D(divergence_velocity, coord.x - 1.0f, coord.y, coord.z);
    float4 center = tex3D(divergence_velocity, coord.x, coord.y, coord.z);
    float4 east =   tex3D(divergence_velocity, coord.x + 1.0f, coord.y, coord.z);
    float4 north =  tex3D(divergence_velocity, coord.x, coord.y + 1.0f, coord.z);
    float4 far =    tex3D(divergence_velocity, coord.x, coord.y, coord.z + 1.0f);

    float diff_ew = east.x - west.x;
    float diff_ns = north.y - south.y;
    float diff_fn = far.z - near.z;

    // Handle boundary problem
    if (x >= volume_size.x - 1)
        diff_ew = -center.x - west.x;

    if (x <= 0)
        diff_ew = east.x + center.x;

    if (y >= volume_size.y - 1)
        diff_ns = -center.y - south.y;

    if (y <= 0)
        diff_ns = north.y + center.y;

    if (z >= volume_size.z - 1)
        diff_fn = -center.z - near.z;

    if (z <= 0)
        diff_fn = far.z + center.z;

    float div = half_inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    ushort2 result = make_ushort2(0, __float2half_rn(div));
    surf3Dwrite(result, divergence_dest, x * sizeof(ushort2), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ComputeResidualPackedDiagnosisKernel(float inverse_h_square,
                                                     int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float  near =   tex3D(diagnosis_source, coord.x, coord.y, coord.z - 1.0f).x;
    float  south =  tex3D(diagnosis_source, coord.x, coord.y - 1.0f, coord.z).x;
    float  west =   tex3D(diagnosis_source, coord.x - 1.0f, coord.y, coord.z).x;
    float2 center = tex3D(diagnosis_source, coord.x, coord.y, coord.z);
    float  east =   tex3D(diagnosis_source, coord.x + 1.0f, coord.y, coord.z).x;
    float  north =  tex3D(diagnosis_source, coord.x, coord.y + 1.0f, coord.z).x;
    float  far =    tex3D(diagnosis_source, coord.x, coord.y, coord.z + 1.0f).x;
    float  b_center = center.y;

    if (coord.y == volume_size.y - 1)
        north = center.x;

    if (coord.y == 0)
        south = center.x;

    if (coord.x == volume_size.x - 1)
        east = center.x;

    if (coord.x == 0)
        west = center.x;

    if (coord.z == volume_size.z - 1)
        far = center.x;

    if (coord.z == 0)
        near = center.x;

    float v = b_center -
        (north + south + east + west + far + near - 6.0 * center.x) *
        inverse_h_square;
    surf3Dwrite(fabsf(v), diagnosis, x * sizeof(float), y, z,
                cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel(float minus_square_cell_size,
                                       float omega_over_beta, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float near =              tex3D(jacobi_packed, x, y, z - 1.0f).x;
    float south =             tex3D(jacobi_packed, x, y - 1.0f, z).x;
    float west =              tex3D(jacobi_packed, x - 1.0f, y, z).x;
    float2 packed_center =    tex3D(jacobi_packed, x, y, z);
    float east =              tex3D(jacobi_packed, x + 1.0f, y, z).x;
    float north =             tex3D(jacobi_packed, x, y + 1.0f, z).x;
    float far =               tex3D(jacobi_packed, x, y, z + 1.0f).x;

    float u = omega_over_beta * 3.0f * packed_center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        packed_center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u),
                               __float2half_rn(packed_center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel_smem_branch(float minus_square_cell_size,
                                                  float omega_over_beta,
                                                  int3 volume_size)
{
    __shared__ float2 cached_block[1000];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int bw = blockDim.x + 2;
    int bh = blockDim.y + 2;

    int index = (threadIdx.z + 1) * bw * bh + (threadIdx.y + 1) * bw +
        threadIdx.x + 1;
    cached_block[index] = tex3D(jacobi_packed, x, y, z);

    if (threadIdx.x == 0)
        cached_block[index - 1] = x == 0 ?                       cached_block[index] : tex3D(jacobi_packed, x - 1, y, z);

    if (threadIdx.x == blockDim.x - 1)
        cached_block[index + 1] = x == volume_size.x - 1 ?       cached_block[index] : tex3D(jacobi_packed, x + 1, y, z);

    if (threadIdx.y == 0)
        cached_block[index - bw] = y == 0 ?                      cached_block[index] : tex3D(jacobi_packed, x, y - 1, z);

    if (threadIdx.y == blockDim.y - 1)
        cached_block[index + bw] = y == volume_size.y - 1 ?      cached_block[index] : tex3D(jacobi_packed, x, y + 1, z);

    if (threadIdx.z == 0)
        cached_block[index - bw * bh] = z == 0 ?                 cached_block[index] : tex3D(jacobi_packed, x, y, z - 1);

    if (threadIdx.z == blockDim.z - 1)
        cached_block[index + bw * bh] = z == volume_size.z - 1 ? cached_block[index] : tex3D(jacobi_packed, x, y, z + 1);

    __syncthreads();

    float  near =   cached_block[index - bw * bh].x;
    float  south =  cached_block[index - bw].x;
    float  west =   cached_block[index - 1].x;
    float2 center = cached_block[index];
    float  east =   cached_block[index + 1].x;
    float  north =  cached_block[index + bw].x;
    float  far =    cached_block[index + bw * bh].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel_smem_assist_thread(
    float minus_square_cell_size, float omega_over_beta, int3 volume_size)
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
    cached_block[index] = tex3D(jacobi_packed, x, y, z);

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
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 1: {
            // south
            inner = (threadIdx.y + 1) * slice_stride + threadIdx.x + 1;
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y - 1;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
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
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
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
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        case 6:
            // north
            inner = (threadIdx.y + 1) * slice_stride + (blockDim.y + 1) * bw +
                threadIdx.x + 1;
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y + blockDim.y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        case 7:
            // far
            inner = (blockDim.z + 1) * slice_stride + (threadIdx.y + 1) * bw +
                threadIdx.x + 1;
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + blockDim.z;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
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
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel_smem_faces_assist_thread(
    float minus_square_cell_size, float omega_over_beta, int3 volume_size)
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

    cached_block[index] = tex3D(jacobi_packed, x, y, z);

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
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 1: {
            // south
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y - 1;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_face_xzy0[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 2: {
            // west
            inner_x = blockIdx.x * blockDim.x - 1;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_face_yzx0[blockDim.y * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 5:
            // east
            inner_x = blockIdx.x * blockDim.x + blockDim.x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_face_yzx1[blockDim.y * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        case 6:
            // north
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y + blockDim.y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_face_xzy1[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        case 7:
            // far
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + blockDim.z;
            cached_face_xyz1[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
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
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel_smem_dedicated_assist_thread(
    float minus_square_cell_size, float omega_over_beta, int3 volume_size)
{
    __shared__ float2 cached_block[1000];

    int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
    int z = blockIdx.z * (blockDim.z - 2) + threadIdx.z - 1;

    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;

    cached_block[index] = tex3D(jacobi_packed, x, y, z);

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
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel_smem_no_halo_storage(
    float minus_square_cell_size, float omega_over_beta, int3 volume_size)
{
    __shared__ float2 cached_block[512];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;

    cached_block[index] = tex3D(jacobi_packed, x, y, z);
    __syncthreads();

    float center = cached_block[index].x;
    float near =  threadIdx.z == 0 ?              (z == 0 ?                 center : tex3D(jacobi_packed, x, y, z - 1.0f).x) : cached_block[index - blockDim.x * blockDim.y].x;
    float south = threadIdx.y == 0 ?              (y == 0 ?                 center : tex3D(jacobi_packed, x, y - 1.0f, z).x) : cached_block[index - blockDim.x].x;
    float west =  threadIdx.x == 0 ?              (x == 0 ?                 center : tex3D(jacobi_packed, x - 1.0f, y, z).x) : cached_block[index - 1].x;
    float east =  threadIdx.x == blockDim.x - 1 ? (x == volume_size.x - 1 ? center : tex3D(jacobi_packed, x + 1.0f, y, z).x) : cached_block[index + 1].x;
    float north = threadIdx.y == blockDim.y - 1 ? (y == volume_size.y - 1 ? center : tex3D(jacobi_packed, x, y + 1.0f, z).x) : cached_block[index + blockDim.x].x;
    float far =   threadIdx.z == blockDim.z - 1 ? (z == volume_size.z - 1 ? center : tex3D(jacobi_packed, x, y, z + 1.0f).x) : cached_block[index + blockDim.x * blockDim.y].x;

    float b_center = cached_block[index].y;

    float u = omega_over_beta * 3.0f * center +
        (west + east + south + north + far + near + minus_square_cell_size *
        b_center) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(b_center));

    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void RoundPassedKernel(int* dest_array, int round, int x)
{
    dest_array[0] = x * x - round * round;
}

__global__ void SubstractGradientPureKernel(float gradient_scale,
                                            int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float near =   tex3D(gradient_packed, coord.x, coord.y, coord.z - 1.0f).x;
    float south =  tex3D(gradient_packed, coord.x, coord.y - 1.0f, coord.z).x;
    float west =   tex3D(gradient_packed, coord.x - 1.0f, coord.y, coord.z).x;
    float center = tex3D(gradient_packed, coord.x, coord.y, coord.z).x;
    float east =   tex3D(gradient_packed, coord.x + 1.0f, coord.y, coord.z).x;
    float north =  tex3D(gradient_packed, coord.x, coord.y + 1.0f, coord.z).x;
    float far =    tex3D(gradient_packed, coord.x, coord.y, coord.z + 1.0f).x;

    float diff_ew = east - west;
    float diff_ns = north - south;
    float diff_fn = far - near;

    // Handle boundary problem
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);
    if (x >= volume_size.x - 1)
        mask.x = 0;

    if (x <= 0)
        mask.x = 0;

    if (y >= volume_size.y - 1)
        mask.y = 0;

    if (y <= 0)
        mask.y = 0;

    if (z >= volume_size.z - 1)
        mask.z = 0;

    if (z <= 0)
        mask.z = 0;

    ushort4 raw;
    surf3Dread(&raw, gradient_velocity, x * sizeof(ushort4), y, z);
    float3 old_v = make_float3(__half2float(raw.x), __half2float(raw.y),
                               __half2float(raw.z));
    float3 grad = make_float3(diff_ew, diff_ns, diff_fn) * gradient_scale;
    float3 new_v = old_v - grad;
    float3 result = mask * new_v; // Velocity goes to 0 when hit ???
    raw = make_ushort4(__float2half_rn(result.x), __float2half_rn(result.y),
                       __float2half_rn(result.z), 0);
    surf3Dwrite(raw, gradient_velocity, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAdvectPure(cudaArray_t dest_array, cudaArray_t velocity_array,
                      cudaArray_t source_array, float time_step,
                      float dissipation, int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_velocity, velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    advect_source.normalized = false;
    advect_source.filterMode = cudaFilterModeLinear;
    advect_source.addressMode[0] = cudaAddressModeClamp;
    advect_source.addressMode[1] = cudaAddressModeClamp;
    advect_source.addressMode[2] = cudaAddressModeClamp;
    advect_source.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectPureKernel<<<grid, block>>>(time_step, dissipation, volume_size);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvectVelocityPure(cudaArray_t dest_array,
                              cudaArray_t velocity_array,
                              float time_step, float dissipation,
                              int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_velocity, velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectVelocityPureKernel<<<grid, block>>>(time_step, dissipation,
                                              volume_size);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchApplyBuoyancyPure(cudaArray* dest_array, cudaArray* velocity_array,
                             cudaArray* temperature_array, float time_step,
                             float ambient_temperature, float accel_factor,
                             float gravity, int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&buoyancy_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
    buoyancy_velocity.normalized = false;
    buoyancy_velocity.filterMode = cudaFilterModePoint;
    buoyancy_velocity.addressMode[0] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[1] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[2] = cudaAddressModeClamp;
    buoyancy_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&buoyancy_velocity, velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, temperature_array);
    buoyancy_temperature.normalized = false;
    buoyancy_temperature.filterMode = cudaFilterModePoint;
    buoyancy_temperature.addressMode[0] = cudaAddressModeClamp;
    buoyancy_temperature.addressMode[1] = cudaAddressModeClamp;
    buoyancy_temperature.addressMode[2] = cudaAddressModeClamp;
    buoyancy_temperature.channelDesc = desc;

    result = cudaBindTextureToArray(&buoyancy_temperature,
                                    temperature_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ApplyBuoyancyPureKernel<<<grid, block>>>(time_step, ambient_temperature,
                                             accel_factor, gravity,
                                             volume_size);

    cudaUnbindTexture(&buoyancy_temperature);
    cudaUnbindTexture(&buoyancy_velocity);
}

void LaunchApplyImpulsePure(cudaArray* dest_array, cudaArray* original_array,
                            float3 center_point, float3 hotspot, float radius,
                            float value, int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&impulse_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, original_array);
    impulse_original.normalized = false;
    impulse_original.filterMode = cudaFilterModeLinear;
    impulse_original.addressMode[0] = cudaAddressModeClamp;
    impulse_original.addressMode[1] = cudaAddressModeClamp;
    impulse_original.addressMode[2] = cudaAddressModeClamp;
    impulse_original.channelDesc = desc;

    result = cudaBindTextureToArray(&impulse_original, original_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(128, 2, 1);
    dim3 grid(volume_size.x / block.x, 1, volume_size.z / block.z);
    ApplyImpulsePureKernel<<<grid, block>>>(center_point, hotspot, radius,
                                            value, volume_size);

    cudaUnbindTexture(&impulse_original);
}

void LaunchComputeDivergencePure(cudaArray* dest_array,
                                 cudaArray* velocity_array,
                                 float half_inverse_cell_size, int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&divergence_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
    divergence_velocity.normalized = false;
    divergence_velocity.filterMode = cudaFilterModePoint;
    divergence_velocity.addressMode[0] = cudaAddressModeClamp;
    divergence_velocity.addressMode[1] = cudaAddressModeClamp;
    divergence_velocity.addressMode[2] = cudaAddressModeClamp;
    divergence_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&divergence_velocity, velocity_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeDivergencePureKernel<<<grid, block>>>(half_inverse_cell_size,
                                                 volume_size);

    cudaUnbindTexture(&divergence_velocity);
}

void LaunchComputeResidualPackedDiagnosis(cudaArray* dest_array,
                                          cudaArray* source_array,
                                          float inverse_h_square,
                                          int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&diagnosis, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    diagnosis_source.normalized = false;
    diagnosis_source.filterMode = cudaFilterModePoint;
    diagnosis_source.addressMode[0] = cudaAddressModeClamp;
    diagnosis_source.addressMode[1] = cudaAddressModeClamp;
    diagnosis_source.addressMode[2] = cudaAddressModeClamp;
    diagnosis_source.channelDesc = desc;

    result = cudaBindTextureToArray(&diagnosis_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeResidualPackedDiagnosisKernel<<<grid, block>>>(inverse_h_square,
                                                          volume_size);

    cudaUnbindTexture(&diagnosis_source);
}

void LaunchDampedJacobiPure(cudaArray* dest_array, cudaArray* source_array,
                            float minus_square_cell_size, float omega_over_beta,
                            int3 volume_size, BlockArrangement* ba)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&jacobi, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    jacobi_packed.normalized = false;
    jacobi_packed.filterMode = cudaFilterModePoint;
    jacobi_packed.addressMode[0] = cudaAddressModeClamp;
    jacobi_packed.addressMode[1] = cudaAddressModeClamp;
    jacobi_packed.addressMode[2] = cudaAddressModeClamp;
    jacobi_packed.channelDesc = desc;

    result = cudaBindTextureToArray(&jacobi_packed, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    bool use_smem = false;
    if (use_smem) {
        dim3 block(8, 8, 8);
        dim3 grid((volume_size.x + block.x - 1) / block.x,
                  (volume_size.x + block.y - 1) / block.y,
                  (volume_size.x + block.z - 1) / block.z);
        DampedJacobiPureKernel_smem_assist_thread<<<grid, block>>>(
            minus_square_cell_size, omega_over_beta, volume_size);
    } else {
        dim3 block;
        dim3 grid;
        ba->Arrange(&block, &grid, volume_size);
        DampedJacobiPureKernel<<<grid, block>>>(minus_square_cell_size,
                                                omega_over_beta, volume_size);
    }
    

    cudaUnbindTexture(&jacobi_packed);
}

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
}

void LaunchSubstractGradientPure(cudaArray* dest_array, cudaArray* packed_array,
                                 float gradient_scale, int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&gradient_velocity, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, packed_array);
    gradient_packed.normalized = false;
    gradient_packed.filterMode = cudaFilterModePoint;
    gradient_packed.addressMode[0] = cudaAddressModeClamp;
    gradient_packed.addressMode[1] = cudaAddressModeClamp;
    gradient_packed.addressMode[2] = cudaAddressModeClamp;
    gradient_packed.channelDesc = desc;

    result = cudaBindTextureToArray(&gradient_packed, packed_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    SubstractGradientPureKernel<<<grid, block>>>(gradient_scale, volume_size);

    cudaUnbindTexture(&gradient_packed);
}
