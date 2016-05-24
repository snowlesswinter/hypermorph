#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> curl_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> curl_velocity;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> curl_curl;

__global__ void ComputeCurlStaggeredKernel(uint3 volume_size,
                                           float inverse_cell_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 result;
    if (x < volume_size.x - 1 && y < volume_size.y - 1 && z < volume_size.z - 1) {
        if (x > 0 && y > 0 && z > 0) {
            float4 v =   tex3D(curl_velocity, coord.x, coord.y, coord.z);
            float4 v_x = tex3D(curl_velocity, coord.x - 1.0f, coord.y, coord.z);
            float4 v_y = tex3D(curl_velocity, coord.x, coord.y - 1.0f, coord.z);
            float4 v_z = tex3D(curl_velocity, coord.x, coord.y, coord.z - 1.0f);
            result.x = inverse_cell_size * (v.z - v_y.z - v.y + v_z.y);
            result.y = inverse_cell_size * (v.x - v_z.x - v.z + v_x.z);
            result.z = inverse_cell_size * (v.y - v_x.y - v.x + v_y.x);
        } else if (x == 0) {
            result = make_float3(
                tex3D(curl_curl, coord.x + 1.0f, coord.y, coord.z));
        } else if (y == 0) {
            result = make_float3(
                tex3D(curl_curl, coord.x, coord.y + 1.0f, coord.z));
        } else {
            result = make_float3(
                tex3D(curl_curl, coord.x, coord.y, coord.z + 1.0f));
        }
    } else if (x == volume_size.x - 1) {
        result =
            make_float3(tex3D(curl_curl, coord.x - 1.0f, coord.y, coord.z));
    } else if (y == volume_size.y - 1) {
        result =
            make_float3(tex3D(curl_curl, coord.x, coord.y - 1.0f, coord.z));
    } else {
        result =
            make_float3(tex3D(curl_curl, coord.x, coord.y, coord.z - 1.0f));
    }

    ushort4 raw = make_ushort4(__float2half_rn(result.x),
                               __float2half_rn(result.y),
                               __float2half_rn(result.z),
                               0);
    surf3Dwrite(result, curl_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchComputeCurlStaggered(cudaArray* dest_array,
                                cudaArray* velocity_array,
                                cudaArray* curl_array,
                                float inverse_cell_size, uint3 volume_size,
                                BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&curl_dest, dest_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&curl_velocity, dest_array,
                                      false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_curl = BindHelper::Bind(&curl_curl, curl_array,
                                      false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_curl.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeCurlStaggeredKernel<<<grid, block>>>(volume_size, inverse_cell_size);
}
