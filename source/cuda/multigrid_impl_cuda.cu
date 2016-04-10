#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

surface<void, cudaTextureType3D> residual_dest;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> residual_source;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_coarse;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_fine;
surface<void, cudaTextureType3D> prolongate_pure_dest;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_pure_coarse;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_pure_fine;
surface<void, cudaTextureType3D> guess_dest;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> guess_source;
surface<void, cudaTextureType3D> restrict_residual_dest;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> restrict_residual_source;
surface<void, cudaTextureType3D> restrict_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> restrict_source;

__global__ void ComputeResidualPackedPureKernel(float inverse_h_square,
                                                int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float  near =   tex3D(residual_source, coord.x, coord.y, coord.z - 1.0f).x;
    float  south =  tex3D(residual_source, coord.x, coord.y - 1.0f, coord.z).x;
    float  west =   tex3D(residual_source, coord.x - 1.0f, coord.y, coord.z).x;
    float2 center = tex3D(residual_source, coord.x, coord.y, coord.z);
    float  east =   tex3D(residual_source, coord.x + 1.0f, coord.y, coord.z).x;
    float  north =  tex3D(residual_source, coord.x, coord.y + 1.0f, coord.z).x;
    float  far =    tex3D(residual_source, coord.x, coord.y, coord.z + 1.0f).x;
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
    ushort raw = __float2half_rn(v);
    surf3Dwrite(raw, residual_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ProlongatePackedKernel(float4* out_data,
                                       int num_of_blocks_per_slice,
                                       int slice_stride, int3 volume_size)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;

    int x = threadIdx.z * blockDim.x + threadIdx.x;
    int z = block_offset / num_of_blocks_per_slice;
    int y = (block_offset - z * num_of_blocks_per_slice) * blockDim.y +
        threadIdx.y;

    int index = slice_stride * z + volume_size.x * y + x;

    float3 c = make_float3(x, y, z);
    c *= 0.5f;

    int odd_x = x - ((x >> 1) << 1);
    int odd_y = y - ((y >> 1) << 1);
    int odd_z = z - ((z >> 1) << 1);

    float t_x = -1.0f * (1 - odd_x) * 0.08333333f;
    float t_y = -1.0f * (1 - odd_y) * 0.08333333f;
    float t_z = -1.0f * (1 - odd_z) * 0.08333333f;

    float3 t_c = make_float3(c.x + t_x, c.y + t_y, c.z + t_z);
    float4 result_float = tex3D(prolongate_coarse, t_c.x, t_c.y, t_c.z);

    float3 f_coord = make_float3(float(x) + 0.5f, float(y) + 0.5f,
                                 float(z) + 0.5f);

    float4 original = tex3D(prolongate_fine, f_coord.x, f_coord.y, f_coord.z);
    float4 result = make_float4(original.x + result_float.x, original.y, 0, 0);

    out_data[index] = result;
}

__global__ void ProlongatePackedPureKernel(int3 volume_size)
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
    float2 result_float = tex3D(prolongate_pure_coarse, t_c.x, t_c.y, t_c.z);

    float3 f_coord = make_float3(x, y, z) + 0.5f;

    float2 original = tex3D(prolongate_pure_fine, f_coord.x, f_coord.y, f_coord.z);
    float2 result = make_float2(original.x + result_float.x, original.y);

    ushort2 raw = make_ushort2(__float2half_rn(result.x),
                               __float2half_rn(result.y));
    surf3Dwrite(raw, prolongate_pure_dest, x * sizeof(ushort2), y, z,
                cudaBoundaryModeTrap);
}

__global__ void RelaxWithZeroGuessPackedPureKernel(
    float alpha_omega_over_beta, float one_minus_omega, float minus_h_square,
    float omega_times_inverse_beta, int3 volume_size)
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

    if (coord.y == volume_size.y - 1)
        north = b_center;

    if (coord.y == 0)
        south = b_center;

    if (coord.x == volume_size.x - 1)
        east = b_center;

    if (coord.x == 0)
        west = b_center;

    if (coord.z == volume_size.z - 1)
        far = b_center;

    if (coord.z == 0)
        near = b_center;

    float v = one_minus_omega * (alpha_omega_over_beta * b_center) +
        (alpha_omega_over_beta * (north + south + east + west + far + near) +
        minus_h_square * b_center) * omega_times_inverse_beta;

    ushort2 raw = make_ushort2(__float2half_rn(v), __float2half_rn(b_center));
    surf3Dwrite(raw, guess_dest, x * sizeof(ushort2), y, z,
                cudaBoundaryModeTrap); 
}

__global__ void RestrictPackedPureKernel(int3 volume_size_fine)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    float c1 = 0.015625f;
    float c2 = 0.03125f;
    float c4 = 0.0625f;
    float c8 = 0.125f;

    // Changing the order of the following voxel-fetching code will NOT affect
    // the performance of this kernel.
    float4 north_east_near =      c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y + 1.0f, coord.z - 1.0f);
    float4 north_center_near =    c2 * tex3D(restrict_source, coord.x,        coord.y + 1.0f, coord.z - 1.0f);
    float4 north_west_near =      c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y + 1.0f, coord.z - 1.0f);
    float4 center_east_near =     c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y,        coord.z - 1.0f);
    float4 center_center_near =   c4 * tex3D(restrict_source, coord.x,        coord.y,        coord.z - 1.0f);
    float4 center_west_near =     c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y,        coord.z - 1.0f);
    float4 south_east_near =      c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y - 1.0f, coord.z - 1.0f);
    float4 south_center_near =    c2 * tex3D(restrict_source, coord.x,        coord.y - 1.0f, coord.z - 1.0f);
    float4 south_west_near =      c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y - 1.0f, coord.z - 1.0f);

    float4 north_east_center =    c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y + 1.0f, coord.z);
    float4 north_center_center =  c4 * tex3D(restrict_source, coord.x,        coord.y + 1.0f, coord.z);
    float4 north_west_center =    c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y + 1.0f, coord.z);
    float4 center_east_center =   c4 * tex3D(restrict_source, coord.x + 1.0f, coord.y,        coord.z);
    float4 center_center_center = c8 * tex3D(restrict_source, coord.x,        coord.y,        coord.z);
    float4 center_west_center =   c4 * tex3D(restrict_source, coord.x - 1.0f, coord.y,        coord.z);
    float4 south_east_center =    c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y - 1.0f, coord.z);
    float4 south_center_center =  c4 * tex3D(restrict_source, coord.x,        coord.y - 1.0f, coord.z);
    float4 south_west_center =    c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y - 1.0f, coord.z);

    float4 north_east_far =       c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y + 1.0f, coord.z + 1.0f);
    float4 north_center_far =     c2 * tex3D(restrict_source, coord.x,        coord.y + 1.0f, coord.z + 1.0f);
    float4 north_west_far =       c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y + 1.0f, coord.z + 1.0f);
    float4 center_east_far =      c2 * tex3D(restrict_source, coord.x + 1.0f, coord.y,        coord.z + 1.0f);
    float4 center_center_far =    c4 * tex3D(restrict_source, coord.x,        coord.y,        coord.z + 1.0f);
    float4 center_west_far =      c2 * tex3D(restrict_source, coord.x - 1.0f, coord.y,        coord.z + 1.0f);
    float4 south_east_far =       c1 * tex3D(restrict_source, coord.x + 1.0f, coord.y - 1.0f, coord.z + 1.0f);
    float4 south_center_far =     c2 * tex3D(restrict_source, coord.x,        coord.y - 1.0f, coord.z + 1.0f);
    float4 south_west_far =       c1 * tex3D(restrict_source, coord.x - 1.0f, coord.y - 1.0f, coord.z + 1.0f);

    float3 tex_size = make_float3(volume_size_fine) - 1.001f;
    float scale = 0.5f;

    if (coord.x > tex_size.x) {
        center_east_center = center_center_center;
    }

    if (coord.x < 1.0001f) { 
        center_west_center = scale * center_center_center;
    }

    if (coord.z > tex_size.z) {
        center_center_far = scale * center_center_center;
    }

    if (coord.z < 1.0001f) {
        center_center_near = scale * center_center_center;
    }

    if (coord.y > tex_size.y) {
        north_center_center = scale * center_center_center;
    }

    if (coord.y < 1.0001f) {
        south_center_center = scale * center_center_center;
    }

    // Pass 2: 1-center cells.
    if (coord.x > tex_size.x) {
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

    if (coord.z > tex_size.z) {
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

    if (coord.y > tex_size.y) {
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
    if (coord.x > tex_size.x) {
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

    if (coord.z > tex_size.z) {
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

    if (coord.y > tex_size.y) {
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

    float4 result =
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

    ushort4 raw = make_ushort4(__float2half_rn(result.x),
                               __float2half_rn(result.y),
                               0, 0);
    surf3Dwrite(raw, restrict_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void RestrictResidualPackedPureKernel(int3 volume_size_fine)
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

    float3 tex_size = make_float3(volume_size_fine) - 1.001f;
    float scale = 0.5f;

    if (coord.x > tex_size.x) {
        center_east_center = center_center_center;
    }

    if (coord.x < 1.0001f) { 
        center_west_center = scale * center_center_center;
    }

    if (coord.z > tex_size.z) {
        center_center_far = scale * center_center_center;
    }

    if (coord.z < 1.0001f) {
        center_center_near = scale * center_center_center;
    }

    if (coord.y > tex_size.y) {
        north_center_center = scale * center_center_center;
    }

    if (coord.y < 1.0001f) {
        south_center_center = scale * center_center_center;
    }

    // Pass 2: 1-center cells.
    if (coord.x > tex_size.x) {
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

    if (coord.z > tex_size.z) {
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

    if (coord.y > tex_size.y) {
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
    if (coord.x > tex_size.x) {
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

    if (coord.z > tex_size.z) {
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

    if (coord.y > tex_size.y) {
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
    surf3Dwrite(raw, restrict_residual_dest, x * sizeof(ushort2), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchComputeResidualPackedPure(cudaArray* dest_array,
                                     cudaArray* source_array,
                                     float inverse_h_square,
                                     int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&residual_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    residual_source.normalized = false;
    residual_source.filterMode = cudaFilterModePoint;
    residual_source.addressMode[0] = cudaAddressModeClamp;
    residual_source.addressMode[1] = cudaAddressModeClamp;
    residual_source.addressMode[2] = cudaAddressModeClamp;
    residual_source.channelDesc = desc;

    result = cudaBindTextureToArray(&residual_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeResidualPackedPureKernel<<<grid, block>>>(inverse_h_square,
                                                     volume_size);

    cudaUnbindTexture(&residual_source);
}

void LaunchProlongatePacked(float4* dest_array, cudaArray* coarse_array,
                            cudaArray* fine_array, int3 volume_size_fine)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, coarse_array);
    prolongate_coarse.normalized = false;
    prolongate_coarse.filterMode = cudaFilterModeLinear;
    prolongate_coarse.addressMode[0] = cudaAddressModeClamp;
    prolongate_coarse.addressMode[1] = cudaAddressModeClamp;
    prolongate_coarse.addressMode[2] = cudaAddressModeClamp;
    prolongate_coarse.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&prolongate_coarse,
                                                coarse_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, fine_array);
    prolongate_fine.normalized = false;

    // TODO: Disabling the linear filter mode may slightly speed up the kernel.
    prolongate_fine.filterMode = cudaFilterModeLinear;
    prolongate_fine.addressMode[0] = cudaAddressModeClamp;
    prolongate_fine.addressMode[1] = cudaAddressModeClamp;
    prolongate_fine.addressMode[2] = cudaAddressModeClamp;
    prolongate_fine.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_fine, fine_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    int3 volume_size = volume_size_fine;
    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    ProlongatePackedKernel<<<grid, block>>>(dest_array, num_of_blocks_per_slice,
                                            slice_stride, volume_size);

    cudaUnbindTexture(&prolongate_fine);
    cudaUnbindTexture(&prolongate_coarse);
}

void LaunchProlongatePackedPure(cudaArray* dest_array, cudaArray* coarse_array,
                                cudaArray* fine_array, int3 volume_size_fine)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&prolongate_pure_dest,
                                                dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, coarse_array);
    prolongate_pure_coarse.normalized = false;
    prolongate_pure_coarse.filterMode = cudaFilterModeLinear;
    prolongate_pure_coarse.addressMode[0] = cudaAddressModeClamp;
    prolongate_pure_coarse.addressMode[1] = cudaAddressModeClamp;
    prolongate_pure_coarse.addressMode[2] = cudaAddressModeClamp;
    prolongate_pure_coarse.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_pure_coarse, coarse_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, fine_array);
    prolongate_pure_fine.normalized = false;
    prolongate_pure_fine.filterMode = cudaFilterModeLinear;
    prolongate_pure_fine.addressMode[0] = cudaAddressModeClamp;
    prolongate_pure_fine.addressMode[1] = cudaAddressModeClamp;
    prolongate_pure_fine.addressMode[2] = cudaAddressModeClamp;
    prolongate_pure_fine.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_pure_fine, fine_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size_fine.x / 8);
    dim3 grid(volume_size_fine.x / block.x, volume_size_fine.y / block.y,
              volume_size_fine.z / block.z);

    ProlongatePackedPureKernel<<<grid, block>>>(volume_size_fine);

    cudaUnbindTexture(&prolongate_pure_fine);
    cudaUnbindTexture(&prolongate_pure_coarse);
}

void LaunchRelaxWithZeroGuessPackedPure(cudaArray* dest_array,
                                        cudaArray* source_array,
                                        float alpha_omega_over_beta,
                                        float one_minus_omega,
                                        float minus_h_square,
                                        float omega_times_inverse_beta,
                                        int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&guess_dest, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    guess_source.normalized = false;
    guess_source.filterMode = cudaFilterModePoint;
    guess_source.addressMode[0] = cudaAddressModeClamp;
    guess_source.addressMode[1] = cudaAddressModeClamp;
    guess_source.addressMode[2] = cudaAddressModeClamp;
    guess_source.channelDesc = desc;

    result = cudaBindTextureToArray(&guess_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RelaxWithZeroGuessPackedPureKernel<<<grid, block>>>(
        alpha_omega_over_beta, one_minus_omega, minus_h_square,
        omega_times_inverse_beta, volume_size);

    cudaUnbindTexture(&guess_source);
}

void LaunchRestrictPackedPure(cudaArray* dest_array, cudaArray* source_array,
                              int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&restrict_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    restrict_source.normalized = false;
    restrict_source.filterMode = cudaFilterModeLinear;
    restrict_source.addressMode[0] = cudaAddressModeClamp;
    restrict_source.addressMode[1] = cudaAddressModeClamp;
    restrict_source.addressMode[2] = cudaAddressModeClamp;
    restrict_source.channelDesc = desc;

    result = cudaBindTextureToArray(&restrict_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    int3 volume_size_fine = volume_size * 2;
    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RestrictPackedPureKernel<<<grid, block>>>(volume_size_fine);

    cudaUnbindTexture(&restrict_source);
}

void LaunchRestrictResidualPackedPure(cudaArray* dest_array,
                                      cudaArray* source_array, int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&restrict_residual_dest,
                                                dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    restrict_residual_source.normalized = false;
    restrict_residual_source.filterMode = cudaFilterModeLinear;
    restrict_residual_source.addressMode[0] = cudaAddressModeClamp;
    restrict_residual_source.addressMode[1] = cudaAddressModeClamp;
    restrict_residual_source.addressMode[2] = cudaAddressModeClamp;
    restrict_residual_source.channelDesc = desc;

    result = cudaBindTextureToArray(&restrict_residual_source, source_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    int3 volume_size_fine = volume_size * 2;
    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RestrictResidualPackedPureKernel<<<grid, block>>>(volume_size_fine);

    cudaUnbindTexture(&restrict_residual_source);
}
