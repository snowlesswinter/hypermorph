#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_b;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_fine;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> residual_source;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_coarse;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_fine;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> guess_source;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> restrict_residual_source;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> restrict_source;

__global__ void ComputeResidualPackedKernel(float inverse_h_square)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float  near =   tex3D(residual_source, x, y, z - 1.0f).x;
    float  south =  tex3D(residual_source, x, y - 1.0f, z).x;
    float  west =   tex3D(residual_source, x - 1.0f, y, z).x;
    float2 center = tex3D(residual_source, x, y, z);
    float  east =   tex3D(residual_source, x + 1.0f, y, z).x;
    float  north =  tex3D(residual_source, x, y + 1.0f, z).x;
    float  far =    tex3D(residual_source, x, y, z + 1.0f).x;
    float  b_center = center.y;

    float v = b_center -
        (north + south + east + west + far + near - 6.0f * center.x) *
        inverse_h_square;
    ushort raw = __float2half_rn(v);
    surf3Dwrite(raw, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeResidualKernel(float inverse_h_square)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float near =   tex3D(tex, x, y, z - 1.0f);
    float south =  tex3D(tex, x, y - 1.0f, z);
    float west =   tex3D(tex, x - 1.0f, y, z);
    float center = tex3D(tex, x, y, z);
    float east =   tex3D(tex, x + 1.0f, y, z);
    float north =  tex3D(tex, x, y + 1.0f, z);
    float far =    tex3D(tex, x, y, z + 1.0f);
    float b =      tex3D(tex_b, x, y, z);

    float v = b -
        (north + south + east + west + far + near - 6.0f * center) *
        inverse_h_square;
    ushort r = __float2half_rn(v);
    surf3Dwrite(r, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void ProlongateLinearInterpolationKernel(float overlay)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;

    float result_float = tex3D(prolongate_coarse, coord.x, coord.y, coord.z).x;

    float2 original = tex3D(prolongate_fine, x, y, z);
    float2 result = make_float2(overlay * original.x + result_float,
                                original.y);

    ushort2 raw = make_ushort2(__float2half_rn(result.x),
                               __float2half_rn(result.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void ProlongateLerpKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;

    float coarse = tex3D(tex, coord.x, coord.y, coord.z);
    float fine = tex3D(tex_fine, x, y, z);
    ushort raw = __float2half_rn(fine + coarse);
    surf3Dwrite(raw, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void ProlongateLinearInterpolation2Kernel(float overlay)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 c = make_float3(x, y, z);
    c *= 0.5f;

    int odd_x = x - ((x >> 1) << 1);
    int odd_y = y - ((y >> 1) << 1);
    int odd_z = z - ((z >> 1) << 1);

    float t_x = -1.0f * (1 - odd_x) * 0.08333333f;
    float t_y = -1.0f * (1 - odd_y) * 0.08333333f;
    float t_z = -1.0f * (1 - odd_z) * 0.08333333f;

    float3 t_c = make_float3(c.x + t_x, c.y + t_y, c.z + t_z);
    float result_float = tex3D(prolongate_coarse, t_c.x, t_c.y, t_c.z).x;

    float2 original = tex3D(prolongate_fine, x, y, z);
    float2 result = make_float2(overlay * original.x + result_float,
                                original.y);

    ushort2 raw = make_ushort2(__float2half_rn(result.x),
                               __float2half_rn(result.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void ProlongateFullWeightedKernel(float overlay, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;

    const float c1 = 0.015625f;
    const float c2 = 0.03125f;
    const float c4 = 0.0625f;
    const float c8 = 0.125f;

    float north_east_near =      c1 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y + 0.5f, coord.z - 0.5f).x;
    float north_center_near =    c2 * tex3D(prolongate_coarse, coord.x,        coord.y + 0.5f, coord.z - 0.5f).x;
    float north_west_near =      c1 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y + 0.5f, coord.z - 0.5f).x;
    float center_east_near =     c2 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y,        coord.z - 0.5f).x;
    float center_center_near =   c4 * tex3D(prolongate_coarse, coord.x,        coord.y,        coord.z - 0.5f).x;
    float center_west_near =     c2 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y,        coord.z - 0.5f).x;
    float south_east_near =      c1 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y - 0.5f, coord.z - 0.5f).x;
    float south_center_near =    c2 * tex3D(prolongate_coarse, coord.x,        coord.y - 0.5f, coord.z - 0.5f).x;
    float south_west_near =      c1 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y - 0.5f, coord.z - 0.5f).x;

    float north_east_center =    c2 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y + 0.5f, coord.z).x;
    float north_center_center =  c4 * tex3D(prolongate_coarse, coord.x,        coord.y + 0.5f, coord.z).x;
    float north_west_center =    c2 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y + 0.5f, coord.z).x;
    float center_east_center =   c4 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y,        coord.z).x;
    float center_center_center = c8 * tex3D(prolongate_coarse, coord.x,        coord.y,        coord.z).x;
    float center_west_center =   c4 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y,        coord.z).x;
    float south_east_center =    c2 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y - 0.5f, coord.z).x;
    float south_center_center =  c4 * tex3D(prolongate_coarse, coord.x,        coord.y - 0.5f, coord.z).x;
    float south_west_center =    c2 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y - 0.5f, coord.z).x;

    float north_east_far =       c1 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y + 0.5f, coord.z + 0.5f).x;
    float north_center_far =     c2 * tex3D(prolongate_coarse, coord.x,        coord.y + 0.5f, coord.z + 0.5f).x;
    float north_west_far =       c1 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y + 0.5f, coord.z + 0.5f).x;
    float center_east_far =      c2 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y,        coord.z + 0.5f).x;
    float center_center_far =    c4 * tex3D(prolongate_coarse, coord.x,        coord.y,        coord.z + 0.5f).x;
    float center_west_far =      c2 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y,        coord.z + 0.5f).x;
    float south_east_far =       c1 * tex3D(prolongate_coarse, coord.x + 0.5f, coord.y - 0.5f, coord.z + 0.5f).x;
    float south_center_far =     c2 * tex3D(prolongate_coarse, coord.x,        coord.y - 0.5f, coord.z + 0.5f).x;
    float south_west_far =       c1 * tex3D(prolongate_coarse, coord.x - 0.5f, coord.y - 0.5f, coord.z + 0.5f).x;

    const float scale = 1.0f;

    if (x == volume_size.x - 1) {
        center_east_center = scale * center_center_center;
    }

    if (x == 0) { 
        center_west_center = scale * center_center_center;
    }

    if (z == volume_size.z - 1) {
        center_center_far = scale * center_center_center;
    }

    if (z == 0) {
        center_center_near = scale * center_center_center;
    }

    if (y == volume_size.y - 1) {
        north_center_center = scale * center_center_center;
    }

    if (y == 0) {
        south_center_center = scale * center_center_center;
    }

    // Pass 2: 1-center cells.
    if (x == volume_size.x - 1) {
        center_east_near = scale * center_center_near;
        north_east_center = scale * north_center_center;
        south_east_center = scale * south_center_center;
        center_east_far = scale * center_center_far;
    }

    if (x == 0) {
        center_west_near = scale * center_center_near;
        north_west_center = scale * north_center_center;
        south_west_center = scale * south_center_center;
        center_west_far = scale * center_center_far;
    }

    if (z == volume_size.z - 1) {
        north_center_far = scale * north_center_center;
        center_east_far = scale * center_east_center;
        center_west_far = scale * center_west_center;
        south_center_far = scale * south_center_center;
    }

    if (z == 0) {
        north_center_near = scale * north_center_center;
        center_east_near = scale * center_east_center;
        center_west_near = scale * center_west_center;
        south_center_near = scale * south_center_center;
    }

    if (y == volume_size.y - 1) {
        north_center_near = scale * center_center_near;
        north_east_center = scale * center_east_center;
        north_west_center = scale * center_west_center;
        north_center_far = scale * center_center_far;
    }

    if (y == 0) {
        south_center_near = scale * center_center_near;
        south_east_center = scale * center_east_center;
        south_west_center = scale * center_west_center;
        south_center_far = scale * center_center_far;
    }

    // Pass 3: corner cells.
    if (x == volume_size.x - 1) {
        north_east_near = scale * north_center_near;
        south_east_near = scale * south_center_near;
        north_east_far = scale * north_center_far;
        south_east_far = scale * south_center_far;
    }

    if (x == 0) {
        north_west_near = scale * north_center_near;
        south_west_near = scale * south_center_near;
        north_west_far = scale * north_center_far;
        south_west_far = scale * south_center_far;
    }

    if (z == volume_size.z - 1) {
        north_east_far = scale * north_east_center;
        north_west_far = scale * north_west_center;
        south_east_far = scale * south_east_center;
        south_west_far = scale * south_west_center;
    }

    if (z == 0) {
        north_east_near = scale * north_east_center;
        north_west_near = scale * north_west_center;
        south_east_near = scale * south_east_center;
        south_west_near = scale * south_west_center;
    }

    if (y == volume_size.y - 1) {
        north_east_near = scale * center_east_near;
        north_west_near = scale * center_west_near;
        north_east_far = scale * center_east_far;
        north_west_far = scale * center_west_far;
    }

    if (y == 0) {
        south_east_near = scale * center_east_near;
        south_west_near = scale * center_west_near;
        south_east_far = scale * center_east_far;
        south_west_far = scale * center_west_far;
    }

    float result_float =
        north_east_near +
        north_center_near +
        north_west_near +
        center_east_near +
        center_center_near +
        center_west_near +
        south_east_near +
        south_center_near +
        south_west_near +

        north_east_center +
        north_center_center +
        north_west_center +
        center_west_center +
        center_center_center +
        center_west_center +
        south_east_center +
        south_center_center +
        south_west_center +

        north_east_far +
        north_center_far +
        north_west_far +
        center_east_far +
        center_center_far +
        center_west_far +
        south_east_far +
        south_center_far +
        south_west_far;

    float2 original = tex3D(prolongate_fine, x, y, z);
    float2 result = make_float2(overlay * original.x + result_float,
                                original.y);

    ushort2 raw = make_ushort2(__float2half_rn(result.x),
                               __float2half_rn(result.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void RelaxWithZeroGuessPackedKernel(float alpha_omega_over_beta,
                                               float one_minus_omega,
                                               float minus_h_square,
                                               float omega_times_inverse_beta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float  near =    tex3D(guess_source, coord.x, coord.y, coord.z - 1.0f).y;
    float  south =   tex3D(guess_source, coord.x, coord.y - 1.0f, coord.z).y;
    float  west =    tex3D(guess_source, coord.x - 1.0f, coord.y, coord.z).y;
    float2 center =  tex3D(guess_source, coord.x, coord.y, coord.z);
    float  east =    tex3D(guess_source, coord.x + 1.0f, coord.y, coord.z).y;
    float  north =   tex3D(guess_source, coord.x, coord.y + 1.0f, coord.z).y;
    float  far =     tex3D(guess_source, coord.x, coord.y, coord.z + 1.0f).y;
    float  b_center = center.y;

    float v = one_minus_omega * (alpha_omega_over_beta * b_center) +
        (alpha_omega_over_beta * (north + south + east + west + far + near) +
        minus_h_square * b_center) * omega_times_inverse_beta;

    ushort2 raw = make_ushort2(__float2half_rn(v), __float2half_rn(b_center));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap); 
}

__global__ void RelaxWithZeroGuessKernel(float alpha_omega_over_beta,
                                         float one_minus_omega,
                                         float minus_h_square,
                                         float omega_times_inverse_beta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float near =   tex3D(tex, coord.x, coord.y, coord.z - 1.0f);
    float south =  tex3D(tex, coord.x, coord.y - 1.0f, coord.z);
    float west =   tex3D(tex, coord.x - 1.0f, coord.y, coord.z);
    float center = tex3D(tex, coord.x, coord.y, coord.z);
    float east =   tex3D(tex, coord.x + 1.0f, coord.y, coord.z);
    float north =  tex3D(tex, coord.x, coord.y + 1.0f, coord.z);
    float far =    tex3D(tex, coord.x, coord.y, coord.z + 1.0f);

    float u = one_minus_omega * (alpha_omega_over_beta * center) +
        (alpha_omega_over_beta * (north + south + east + west + far + near) +
        minus_h_square * center) * omega_times_inverse_beta;

    ushort r = __float2half_rn(u);
    surf3Dwrite(r, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void RestrictFullWeightedKernel(uint3 volume_size_fine)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    const float c1 = 0.015625f;
    const float c2 = 0.03125f;
    const float c4 = 0.0625f;
    const float c8 = 0.125f;

    // Changing the order of the following voxel-fetching code will neither
    // affect the cache efficiency nor the performance of this kernel.
    float2 north_east_near =      c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y + 1.0f, coord.z - 1.0f);
    float2 north_center_near =    c2 * tex3D(restrict_source, coord.x,        coord.y + 1.0f, coord.z - 1.0f);
    float2 north_west_near =      c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y + 1.0f, coord.z - 1.0f);
    float2 center_east_near =     c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y,        coord.z - 1.0f);
    float2 center_center_near =   c4 * tex3D(restrict_source, coord.x,        coord.y,        coord.z - 1.0f);
    float2 center_west_near =     c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y,        coord.z - 1.0f);
    float2 south_east_near =      c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y - 1.0f, coord.z - 1.0f);
    float2 south_center_near =    c2 * tex3D(restrict_source, coord.x,        coord.y - 1.0f, coord.z - 1.0f);
    float2 south_west_near =      c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y - 1.0f, coord.z - 1.0f);

    float2 north_east_center =    c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y + 1.0f, coord.z);
    float2 north_center_center =  c4 * tex3D(restrict_source, coord.x,        coord.y + 1.0f, coord.z);
    float2 north_west_center =    c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y + 1.0f, coord.z);
    float2 center_east_center =   c4 * tex3D(restrict_source, coord.x + 1.0f, coord.y,        coord.z);
    float2 center_center_center = c8 * tex3D(restrict_source, coord.x,        coord.y,        coord.z);
    float2 center_west_center =   c4 * tex3D(restrict_source, coord.x - 1.0f, coord.y,        coord.z);
    float2 south_east_center =    c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y - 1.0f, coord.z);
    float2 south_center_center =  c4 * tex3D(restrict_source, coord.x,        coord.y - 1.0f, coord.z);
    float2 south_west_center =    c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y - 1.0f, coord.z);

    float2 north_east_far =       c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y + 1.0f, coord.z + 1.0f);
    float2 north_center_far =     c2 * tex3D(restrict_source, coord.x,        coord.y + 1.0f, coord.z + 1.0f);
    float2 north_west_far =       c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y + 1.0f, coord.z + 1.0f);
    float2 center_east_far =      c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y,        coord.z + 1.0f);
    float2 center_center_far =    c4 * tex3D(restrict_source, coord.x,        coord.y,        coord.z + 1.0f);
    float2 center_west_far =      c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y,        coord.z + 1.0f);
    float2 south_east_far =       c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y - 1.0f, coord.z + 1.0f);
    float2 south_center_far =     c2 * tex3D(restrict_source, coord.x,        coord.y - 1.0f, coord.z + 1.0f);
    float2 south_west_far =       c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y - 1.0f, coord.z + 1.0f);

    const float scale = 0.5f;

    if (coord.x >= volume_size_fine.x - 1) {
        center_east_center = scale * center_center_center;
    }

    if (coord.x < 1.0001f) { 
        center_west_center = scale * center_center_center;
    }

    if (coord.z >= volume_size_fine.z - 1) {
        center_center_far = scale * center_center_center;
    }

    if (coord.z < 1.0001f) {
        center_center_near = scale * center_center_center;
    }

    if (coord.y >= volume_size_fine.y - 1) {
        north_center_center = scale * center_center_center;
    }

    if (coord.y < 1.0001f) {
        south_center_center = scale * center_center_center;
    }

    // Pass 2: 1-center cells.
    if (coord.x >= volume_size_fine.x - 1) {
        center_east_near = scale * center_center_near;
        north_east_center = scale * north_center_center;
        south_east_center = scale * south_center_center;
        center_east_far = scale * center_center_far;
    }

    if (coord.x < 1.0001f) {
        center_west_near = scale * center_center_near;
        north_west_center = scale * north_center_center;
        south_west_center = scale * south_center_center;
        center_west_far = scale * center_center_far;
    }

    if (coord.z >= volume_size_fine.z - 1) {
        north_center_far = scale * north_center_center;
        center_east_far = scale * center_east_center;
        center_west_far = scale * center_west_center;
        south_center_far = scale * south_center_center;
    }

    if (coord.z < 1.0001f) {
        north_center_near = scale * north_center_center;
        center_east_near = scale * center_east_center;
        center_west_near = scale * center_west_center;
        south_center_near = scale * south_center_center;
    }

    if (coord.y >= volume_size_fine.y - 1) {
        north_center_near = scale * center_center_near;
        north_east_center = scale * center_east_center;
        north_west_center = scale * center_west_center;
        north_center_far = scale * center_center_far;
    }

    if (coord.y < 1.0001f) {
        south_center_near = scale * center_center_near;
        south_east_center = scale * center_east_center;
        south_west_center = scale * center_west_center;
        south_center_far = scale * center_center_far;
    }

    // Pass 3: corner cells.
    if (coord.x >= volume_size_fine.x - 1) {
        north_east_near = scale * north_center_near;
        south_east_near = scale * south_center_near;
        north_east_far = scale * north_center_far;
        south_east_far = scale * south_center_far;
    }

    if (coord.x < 1.0001f) {
        north_west_near = scale * north_center_near;
        south_west_near = scale * south_center_near;
        north_west_far = scale * north_center_far;
        south_west_far = scale * south_center_far;
    }

    if (coord.z >= volume_size_fine.z - 1) {
        north_east_far = scale * north_east_center;
        north_west_far = scale * north_west_center;
        south_east_far = scale * south_east_center;
        south_west_far = scale * south_west_center;
    }

    if (coord.z < 1.0001f) {
        north_east_near = scale * north_east_center;
        north_west_near = scale * north_west_center;
        south_east_near = scale * south_east_center;
        south_west_near = scale * south_west_center;
    }

    if (coord.y >= volume_size_fine.y - 1) {
        north_east_near = scale * center_east_near;
        north_west_near = scale * center_west_near;
        north_east_far = scale * center_east_far;
        north_west_far = scale * center_west_far;
    }

    if (coord.y < 1.0001f) {
        south_east_near = scale * center_east_near;
        south_west_near = scale * center_west_near;
        south_east_far = scale * center_east_far;
        south_west_far = scale * center_west_far;
    }

    float2 result =
        north_east_near +
        north_center_near +
        north_west_near +
        center_east_near +
        center_center_near +
        center_west_near +
        south_east_near +
        south_center_near +
        south_west_near +

        north_east_center +
        north_center_center +
        north_west_center +
        center_west_center +
        center_center_center +
        center_west_center +
        south_east_center +
        south_center_center +
        south_west_center +

        north_east_far +
        north_center_far +
        north_west_far +
        center_east_far +
        center_center_far +
        center_west_far +
        south_east_far +
        south_center_far +
        south_west_far;

    ushort2 raw = make_ushort2(__float2half_rn(result.x),
                               __float2half_rn(result.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void RestrictLinearInterpolationKernel(uint3 volume_size_fine)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    float2 result = tex3D(restrict_source, coord.x, coord.y, coord.z);
    ushort2 raw = make_ushort2(__float2half_rn(result.x),
                               __float2half_rn(result.y));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void RestrictResidualFullWeightedKernel(uint3 volume_size_fine)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    float c1 = 0.015625f;
    float c2 = 0.03125f;
    float c4 = 0.0625f;
    float c8 = 0.125f;

    float north_east_near =      c1 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y + 1.0f, coord.z - 1.0f);
    float north_center_near =    c2 * tex3D(restrict_residual_source, coord.x,        coord.y + 1.0f, coord.z - 1.0f);
    float north_west_near =      c1 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y + 1.0f, coord.z - 1.0f);
    float center_east_near =     c2 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y,        coord.z - 1.0f);
    float center_center_near =   c4 * tex3D(restrict_residual_source, coord.x,        coord.y,        coord.z - 1.0f);
    float center_west_near =     c2 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y,        coord.z - 1.0f);
    float south_east_near =      c1 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y - 1.0f, coord.z - 1.0f);
    float south_center_near =    c2 * tex3D(restrict_residual_source, coord.x,        coord.y - 1.0f, coord.z - 1.0f);
    float south_west_near =      c1 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y - 1.0f, coord.z - 1.0f);

    float north_east_center =    c2 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y + 1.0f, coord.z);
    float north_center_center =  c4 * tex3D(restrict_residual_source, coord.x,        coord.y + 1.0f, coord.z);
    float north_west_center =    c2 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y + 1.0f, coord.z);
    float center_east_center =   c4 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y,        coord.z);
    float center_center_center = c8 * tex3D(restrict_residual_source, coord.x,        coord.y,        coord.z);
    float center_west_center =   c4 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y,        coord.z);
    float south_east_center =    c2 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y - 1.0f, coord.z);
    float south_center_center =  c4 * tex3D(restrict_residual_source, coord.x,        coord.y - 1.0f, coord.z);
    float south_west_center =    c2 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y - 1.0f, coord.z);

    float north_east_far =       c1 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y + 1.0f, coord.z + 1.0f);
    float north_center_far =     c2 * tex3D(restrict_residual_source, coord.x,        coord.y + 1.0f, coord.z + 1.0f);
    float north_west_far =       c1 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y + 1.0f, coord.z + 1.0f);
    float center_east_far =      c2 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y,        coord.z + 1.0f);
    float center_center_far =    c4 * tex3D(restrict_residual_source, coord.x,        coord.y,        coord.z + 1.0f);
    float center_west_far =      c2 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y,        coord.z + 1.0f);
    float south_east_far =       c1 * tex3D(restrict_residual_source, coord.x + 1.0f, coord.y - 1.0f, coord.z + 1.0f);
    float south_center_far =     c2 * tex3D(restrict_residual_source, coord.x,        coord.y - 1.0f, coord.z + 1.0f);
    float south_west_far =       c1 * tex3D(restrict_residual_source, coord.x - 1.0f, coord.y - 1.0f, coord.z + 1.0f);

    const float scale = 0.5f;

    if (coord.x >= volume_size_fine.x - 1) {
        center_east_center = scale * center_center_center;
    }

    if (coord.x < 1.0001f) { 
        center_west_center = scale * center_center_center;
    }

    if (coord.z >= volume_size_fine.z - 1) {
        center_center_far = scale * center_center_center;
    }

    if (coord.z < 1.0001f) {
        center_center_near = scale * center_center_center;
    }

    if (coord.y >= volume_size_fine.y - 1) {
        north_center_center = scale * center_center_center;
    }

    if (coord.y < 1.0001f) {
        south_center_center = scale * center_center_center;
    }

    // Pass 2: 1-center cells.
    if (coord.x >= volume_size_fine.x - 1) {
        center_east_near = scale * center_center_near;
        north_east_center = scale * north_center_center;
        south_east_center = scale * south_center_center;
        center_east_far = scale * center_center_far;
    }

    if (coord.x < 1.0001f) {
        center_west_near = scale * center_center_near;
        north_west_center = scale * north_center_center;
        south_west_center = scale * south_center_center;
        center_west_far = scale * center_center_far;
    }

    if (coord.z >= volume_size_fine.z - 1) {
        north_center_far = scale * north_center_center;
        center_east_far = scale * center_east_center;
        center_west_far = scale * center_west_center;
        south_center_far = scale * south_center_center;
    }

    if (coord.z < 1.0001f) {
        north_center_near = scale * north_center_center;
        center_east_near = scale * center_east_center;
        center_west_near = scale * center_west_center;
        south_center_near = scale * south_center_center;
    }

    if (coord.y >= volume_size_fine.y - 1) {
        north_center_near = scale * center_center_near;
        north_east_center = scale * center_east_center;
        north_west_center = scale * center_west_center;
        north_center_far = scale * center_center_far;
    }

    if (coord.y < 1.0001f) {
        south_center_near = scale * center_center_near;
        south_east_center = scale * center_east_center;
        south_west_center = scale * center_west_center;
        south_center_far = scale * center_center_far;
    }

    // Pass 3: corner cells.
    if (coord.x >= volume_size_fine.x - 1) {
        north_east_near = scale * north_center_near;
        south_east_near = scale * south_center_near;
        north_east_far = scale * north_center_far;
        south_east_far = scale * south_center_far;
    }

    if (coord.x < 1.0001f) {
        north_west_near = scale * north_center_near;
        south_west_near = scale * south_center_near;
        north_west_far = scale * north_center_far;
        south_west_far = scale * south_center_far;
    }

    if (coord.z >= volume_size_fine.z - 1) {
        north_east_far = scale * north_east_center;
        north_west_far = scale * north_west_center;
        south_east_far = scale * south_east_center;
        south_west_far = scale * south_west_center;
    }

    if (coord.z < 1.0001f) {
        north_east_near = scale * north_east_center;
        north_west_near = scale * north_west_center;
        south_east_near = scale * south_east_center;
        south_west_near = scale * south_west_center;
    }

    if (coord.y >= volume_size_fine.y - 1) {
        north_east_near = scale * center_east_near;
        north_west_near = scale * center_west_near;
        north_east_far = scale * center_east_far;
        north_west_far = scale * center_west_far;
    }

    if (coord.y < 1.0001f) {
        south_east_near = scale * center_east_near;
        south_west_near = scale * center_west_near;
        south_east_far = scale * center_east_far;
        south_west_far = scale * center_west_far;
    }

    float result =
        north_east_near +
        north_center_near +
        north_west_near +
        center_east_near +
        center_center_near +
        center_west_near +
        south_east_near +
        south_center_near +
        south_west_near +

        north_east_center +
        north_center_center +
        north_west_center +
        center_west_center +
        center_center_center +
        center_west_center +
        south_east_center +
        south_center_center +
        south_west_center +

        north_east_far +
        north_center_far +
        north_west_far +
        center_east_far +
        center_center_far +
        center_west_far +
        south_east_far +
        south_center_far +
        south_west_far;

    ushort2 raw = make_ushort2(0, __float2half_rn(result));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void RestrictResidualLinearInterpolationKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    float result = tex3D(restrict_residual_source, coord.x, coord.y, coord.z);
    ushort2 raw = make_ushort2(0, __float2half_rn(result));
    surf3Dwrite(raw, surf, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void RestrictResidualLerpKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    ushort r = __float2half_rn(tex3D(tex, coord.x, coord.y, coord.z));
    surf3Dwrite(r, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchComputeResidualPacked(cudaArray* dest_array, cudaArray* source_array,
                                 float inverse_h_square, uint3 volume_size,
                                 BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&residual_source, source_array, false,
                                         cudaFilterModePoint,
                                         cudaAddressModeClamp);
    if (bound_source.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);
    ComputeResidualPackedKernel<<<grid, block>>>(inverse_h_square);
}

void LaunchComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                           float cell_size, uint3 volume_size,
                           BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, r) != cudaSuccess)
        return;

    auto bound_u = BindHelper::Bind(&tex, u, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_u.error() != cudaSuccess)
        return;

    auto bound_b = BindHelper::Bind(&tex_b, b, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_b.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);
    ComputeResidualKernel<<<grid, block>>>(1.0f / (cell_size * cell_size));
}

void LaunchProlongate(cudaArray* fine, cudaArray* coarse,
                      uint3 volume_size_fine, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, fine) != cudaSuccess)
        return;

    auto bound_coarse = BindHelper::Bind(&tex, coarse, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_coarse.error() != cudaSuccess)
        return;

    auto bound_fine = BindHelper::Bind(&tex_fine, fine, false,
                                       cudaFilterModePoint,
                                       cudaAddressModeClamp);
    if (bound_fine.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size_fine);
    ProlongateLerpKernel<<<grid, block>>>();
}

void LaunchProlongatePacked(cudaArray* dest_array, cudaArray* coarse_array,
                            cudaArray* fine_array, float overlay,
                            uint3 volume_size_fine, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
        return;

    auto bound_coarse = BindHelper::Bind(&prolongate_coarse, coarse_array,
                                         false, cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_coarse.error() != cudaSuccess)
        return;

    auto bound_fine = BindHelper::Bind(&prolongate_fine, fine_array, false,
                                       cudaFilterModePoint,
                                       cudaAddressModeClamp);
    if (bound_fine.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size_fine);
    ProlongateLinearInterpolationKernel<<<grid, block>>>(overlay);
}

void LaunchRelaxWithZeroGuess(cudaArray* u, cudaArray* b, float cell_size,
                              uint3 volume_size, BlockArrangement* ba)
{
    float alpha_omega_over_beta = -(cell_size * cell_size) * 0.11111111f;
    float one_minus_omega = 0.33333333f;
    float minus_h_square = -(cell_size * cell_size);
    float omega_times_inverse_beta = 0.11111111f;

    if (BindCudaSurfaceToArray(&surf, u) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, b, false, cudaFilterModePoint,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    RelaxWithZeroGuessKernel<<<grid, block>>>(alpha_omega_over_beta,
                                              one_minus_omega,
                                              minus_h_square,
                                              omega_times_inverse_beta);
}

void LaunchRelaxWithZeroGuessPacked(cudaArray* dest_array,
                                    cudaArray* source_array,
                                    float alpha_omega_over_beta,
                                    float one_minus_omega, float minus_h_square,
                                    float omega_times_inverse_beta,
                                    uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&guess_source, source_array, false,
                                         cudaFilterModePoint,
                                         cudaAddressModeClamp);
    if (bound_source.error() != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RelaxWithZeroGuessPackedKernel<<<grid, block>>>(alpha_omega_over_beta,
                                                    one_minus_omega,
                                                    minus_h_square,
                                                    omega_times_inverse_beta);
}

void LaunchRestrictPacked(cudaArray* dest_array, cudaArray* source_array,
                          uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&restrict_source, source_array, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_source.error() != cudaSuccess)
        return;

    uint3 volume_size_fine = volume_size * 2;
    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RestrictLinearInterpolationKernel<<<grid, block>>>(volume_size_fine);
}

void LaunchRestrictResidualPacked(cudaArray* dest_array,
                                  cudaArray* source_array, uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&restrict_residual_source,
                                         source_array, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_source.error() != cudaSuccess)
        return;

    uint3 volume_size_fine = volume_size * 2;
    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RestrictResidualLinearInterpolationKernel<<<grid, block>>>();
}

void LaunchRestrictResidual(cudaArray* b, cudaArray* r, uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, b) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, r, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    uint3 volume_size_fine = volume_size * 2;
    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RestrictResidualLerpKernel<<<grid, block>>>();
}
