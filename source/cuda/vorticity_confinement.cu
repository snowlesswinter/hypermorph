#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> curl_dest_x;
surface<void, cudaSurfaceType3D> curl_dest_y;
surface<void, cudaSurfaceType3D> curl_dest_z;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> curl_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> curl_curl_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> curl_curl_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> curl_curl_z;
surface<void, cudaSurfaceType3D> vort_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> vort_curl;

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
            result.x = tex3D(curl_curl_x, coord.x + 1.0f, coord.y, coord.z);
            result.y = tex3D(curl_curl_y, coord.x + 1.0f, coord.y, coord.z);
            result.z = tex3D(curl_curl_z, coord.x + 1.0f, coord.y, coord.z);
        } else if (y == 0) {
            result.x = tex3D(curl_curl_x, coord.x, coord.y + 1.0f, coord.z);
            result.y = tex3D(curl_curl_y, coord.x, coord.y + 1.0f, coord.z);
            result.z = tex3D(curl_curl_z, coord.x, coord.y + 1.0f, coord.z);
        } else {
            result.x = tex3D(curl_curl_x, coord.x, coord.y, coord.z + 1.0f);
            result.y = tex3D(curl_curl_y, coord.x, coord.y, coord.z + 1.0f);
            result.z = tex3D(curl_curl_z, coord.x, coord.y, coord.z + 1.0f);
        }
    } else if (x == volume_size.x - 1) {
        result.x = tex3D(curl_curl_x, coord.x - 1.0f, coord.y, coord.z);
        result.y = tex3D(curl_curl_y, coord.x - 1.0f, coord.y, coord.z);
        result.z = tex3D(curl_curl_z, coord.x - 1.0f, coord.y, coord.z);
    } else if (y == volume_size.y - 1) {
        result.x = tex3D(curl_curl_x, coord.x, coord.y - 1.0f, coord.z);
        result.y = tex3D(curl_curl_y, coord.x, coord.y - 1.0f, coord.z);
        result.z = tex3D(curl_curl_z, coord.x, coord.y - 1.0f, coord.z);
    } else {
        result.x = tex3D(curl_curl_x, coord.x, coord.y, coord.z - 1.0f);
        result.y = tex3D(curl_curl_y, coord.x, coord.y, coord.z - 1.0f);
        result.z = tex3D(curl_curl_z, coord.x, coord.y, coord.z - 1.0f);
    }

    ushort raw = __float2half_rn(result.x);
    surf3Dwrite(raw, curl_dest_x, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);

    raw = __float2half_rn(result.y);
    surf3Dwrite(raw, curl_dest_y, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);

    raw = __float2half_rn(result.z);
    surf3Dwrite(raw, curl_dest_z, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchComputeCurlStaggered(cudaArray* dest_x, cudaArray* dest_y,
                                cudaArray* dest_z, cudaArray* velocity,
                                cudaArray* curl_x, cudaArray* curl_y,
                                cudaArray* curl_z, float inverse_cell_size,
                                uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&curl_dest_x, dest_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&curl_dest_y, dest_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&curl_dest_z, dest_z) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&curl_velocity, velocity, false,
                                      cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_curl_x = BindHelper::Bind(&curl_curl_x, curl_x, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_x.error() != cudaSuccess)
        return;

    auto bound_curl_y = BindHelper::Bind(&curl_curl_y, curl_y, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_y.error() != cudaSuccess)
        return;

    auto bound_curl_z = BindHelper::Bind(&curl_curl_z, curl_z, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeCurlStaggeredKernel<<<grid, block>>>(volume_size, inverse_cell_size);
}
