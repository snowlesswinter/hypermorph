#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_div;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vort_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vort_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vort_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_conf_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_conf_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_conf_z;

__global__ void ApplyVorticityConfinementStaggeredKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x == 0 || y == 0 || z == 0)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float conf_x = tex3D(tex_conf_x, coord.x - 0.5f, coord.y + 0.5f, coord.z + 0.5f);
    float conf_y = tex3D(tex_conf_y, coord.x + 0.5f, coord.y - 0.5f, coord.z + 0.5f);
    float conf_z = tex3D(tex_conf_z, coord.x + 0.5f, coord.y + 0.5f, coord.z - 0.5f);
    float4 velocity = tex3D(tex_velocity, coord.x, coord.y, coord.z);
    ushort4 result = make_ushort4(__float2half_rn(velocity.x + conf_x),
                                  __float2half_rn(velocity.y + conf_y),
                                  __float2half_rn(velocity.z + conf_z),
                                  0);
    surf3Dwrite(result, surf, x * sizeof(ushort4), y, z, cudaBoundaryModeTrap);
}

__global__ void BuildVorticityConfinementStaggeredKernel(
    float coeff, float cell_size, float half_inverse_cell_size,
    uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    // Calculate the gradient of vorticity.
    float near_vort_x =   tex3D(tex_vort_x, coord.x,        coord.y + 0.5f, coord.z - 0.5f);
    float near_vort_y =   tex3D(tex_vort_y, coord.x + 0.5f, coord.y,        coord.z - 0.5f);
    float near_vort_z =   tex3D(tex_vort_z, coord.x + 0.5f, coord.y + 0.5f, coord.z - 1.0f);
    float near_vort = sqrtf(near_vort_x * near_vort_x + near_vort_y * near_vort_y + near_vort_z * near_vort_z);

    float south_vort_x =  tex3D(tex_vort_x, coord.x,        coord.y - 0.5f, coord.z + 0.5f);
    float south_vort_y =  tex3D(tex_vort_y, coord.x + 0.5f, coord.y - 1.0f, coord.z + 0.5f);
    float south_vort_z =  tex3D(tex_vort_z, coord.x + 0.5f, coord.y - 0.5f, coord.z);
    float south_vort = sqrtf(south_vort_x * south_vort_x + south_vort_y * south_vort_y + south_vort_z * south_vort_z);

    float west_vort_x =   tex3D(tex_vort_x, coord.x - 1.0f, coord.y + 0.5f, coord.z + 0.5f);
    float west_vort_y =   tex3D(tex_vort_y, coord.x - 0.5f, coord.y,        coord.z + 0.5f);
    float west_vort_z =   tex3D(tex_vort_z, coord.x - 0.5f, coord.y + 0.5f, coord.z);
    float west_vort = sqrtf(west_vort_x * west_vort_x + west_vort_y * west_vort_y + west_vort_z * west_vort_z);

    float center_vort_x = tex3D(tex_vort_x, coord.x,        coord.y,        coord.z);
    float center_vort_y = tex3D(tex_vort_y, coord.x,        coord.y,        coord.z);
    float center_vort_z = tex3D(tex_vort_z, coord.x,        coord.y,        coord.z);

    float east_vort_x =   tex3D(tex_vort_x, coord.x + 1.0f, coord.y + 0.5f, coord.z + 0.5f);
    float east_vort_y =   tex3D(tex_vort_y, coord.x + 1.5f, coord.y,        coord.z + 0.5f);
    float east_vort_z =   tex3D(tex_vort_z, coord.x + 1.5f, coord.y + 0.5f, coord.z);
    float east_vort = sqrtf(east_vort_x * east_vort_x + east_vort_y * east_vort_y + east_vort_z * east_vort_z);

    float north_vort_x =  tex3D(tex_vort_x, coord.x,        coord.y + 1.5f, coord.z + 0.5f);
    float north_vort_y =  tex3D(tex_vort_y, coord.x + 0.5f, coord.y + 1.0f, coord.z + 0.5f);
    float north_vort_z =  tex3D(tex_vort_z, coord.x + 0.5f, coord.y + 1.5f, coord.z);
    float north_vort = sqrtf(north_vort_x * north_vort_x + north_vort_y * north_vort_y + north_vort_z * north_vort_z);

    float far_vort_x =    tex3D(tex_vort_x, coord.x,        coord.y + 0.5f, coord.z + 1.5f);
    float far_vort_y =    tex3D(tex_vort_y, coord.x + 0.5f, coord.y,        coord.z + 1.5f);
    float far_vort_z =    tex3D(tex_vort_z, coord.x + 0.5f, coord.y + 0.5f, coord.z + 1.0f);
    float far_vort = sqrtf(far_vort_x * far_vort_x + far_vort_y * far_vort_y + far_vort_z * far_vort_z);

    // Calculate normalized ¦Ç.
    float ¦Ç_x = half_inverse_cell_size * (east_vort - west_vort);
    float ¦Ç_y = half_inverse_cell_size * (north_vort - south_vort);
    float ¦Ç_z = half_inverse_cell_size * (far_vort - near_vort);

    float ¦Ç_mag = sqrtf(¦Ç_x * ¦Ç_x + ¦Ç_y * ¦Ç_y + ¦Ç_z * ¦Ç_z + 0.00001f);
    ¦Ç_x /= ¦Ç_mag;
    ¦Ç_y /= ¦Ç_mag;
    ¦Ç_z /= ¦Ç_mag;

    // Vorticity confinement at the center of the grid.
    float tex_conf_x = coeff * cell_size * (¦Ç_y * center_vort_z - ¦Ç_z * center_vort_y);
    float tex_conf_y = coeff * cell_size * (¦Ç_z * center_vort_x - ¦Ç_x * center_vort_z);
    float tex_conf_z = coeff * cell_size * (¦Ç_x * center_vort_y - ¦Ç_y * center_vort_x);

    ushort result_x = __float2half_rn(tex_conf_x);
    surf3Dwrite(result_x, surf_x, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);

    ushort result_y = __float2half_rn(tex_conf_y);
    surf3Dwrite(result_y, surf_y, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);

    ushort result_z = __float2half_rn(tex_conf_z);
    surf3Dwrite(result_z, surf_z, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

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
            float4 v_near =  tex3D(tex_velocity, coord.x, coord.y, coord.z - 1.0f);
            float4 v_west =  tex3D(tex_velocity, coord.x - 1.0f, coord.y, coord.z);
            float4 v_south = tex3D(tex_velocity, coord.x, coord.y - 1.0f, coord.z);
            float4 v =       tex3D(tex_velocity, coord.x, coord.y, coord.z);

            result.x = inverse_cell_size * (v.z - v_south.z - v.y + v_near.y);
            result.y = inverse_cell_size * (v.x - v_near.x  - v.z + v_west.z);
            result.z = inverse_cell_size * (v.y - v_west.y  - v.x + v_south.x);
        } else if (x == 0) {
            result.x = tex3D(tex_vort_x, coord.x + 1.0f, coord.y, coord.z);
            result.y = tex3D(tex_vort_y, coord.x + 1.0f, coord.y, coord.z);
            result.z = tex3D(tex_vort_z, coord.x + 1.0f, coord.y, coord.z);
        } else if (y == 0) {
            result.x = tex3D(tex_vort_x, coord.x, coord.y + 1.0f, coord.z);
            result.y = tex3D(tex_vort_y, coord.x, coord.y + 1.0f, coord.z);
            result.z = tex3D(tex_vort_z, coord.x, coord.y + 1.0f, coord.z);
        } else {
            result.x = tex3D(tex_vort_x, coord.x, coord.y, coord.z + 1.0f);
            result.y = tex3D(tex_vort_y, coord.x, coord.y, coord.z + 1.0f);
            result.z = tex3D(tex_vort_z, coord.x, coord.y, coord.z + 1.0f);
        }
    } else if (x == volume_size.x - 1) {
        result.x = tex3D(tex_vort_x, coord.x - 1.0f, coord.y, coord.z);
        result.y = tex3D(tex_vort_y, coord.x - 1.0f, coord.y, coord.z);
        result.z = tex3D(tex_vort_z, coord.x - 1.0f, coord.y, coord.z);
    } else if (y == volume_size.y - 1) {
        result.x = tex3D(tex_vort_x, coord.x, coord.y - 1.0f, coord.z);
        result.y = tex3D(tex_vort_y, coord.x, coord.y - 1.0f, coord.z);
        result.z = tex3D(tex_vort_z, coord.x, coord.y - 1.0f, coord.z);
    } else {
        result.x = tex3D(tex_vort_x, coord.x, coord.y, coord.z - 1.0f);
        result.y = tex3D(tex_vort_y, coord.x, coord.y, coord.z - 1.0f);
        result.z = tex3D(tex_vort_z, coord.x, coord.y, coord.z - 1.0f);
    }

    ushort raw_x = __float2half_rn(result.x);
    surf3Dwrite(raw_x, surf_x, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);

    ushort raw_y = __float2half_rn(result.y);
    surf3Dwrite(raw_y, surf_y, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);

    ushort raw_z = __float2half_rn(result.z);
    surf3Dwrite(raw_z, surf_z, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeDivergenceStaggeredKernelForVort(float inverse_cell_size,
                                                        uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float4 base =   tex3D(tex_velocity, coord.x,        coord.y,        coord.z);
    float  east =   tex3D(tex_velocity, coord.x + 1.0f, coord.y,        coord.z).x;
    float  north =  tex3D(tex_velocity, coord.x,        coord.y + 1.0f, coord.z).y;
    float  far =    tex3D(tex_velocity, coord.x,        coord.y,        coord.z + 1.0f).z;

    float diff_ew = east  - base.x;
    float diff_ns = north - base.y;
    float diff_fn = far   - base.z;

    // Handle boundary problem
    if (x >= volume_size.x - 1)
        diff_ew = -base.x;

    if (y >= volume_size.y - 1)
        diff_ns = -base.y;

    if (z >= volume_size.z - 1)
        diff_fn = -base.z;

    float div = inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    ushort result = __float2half_rn(div);
    surf3Dwrite(result, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void DecayVorticesStaggeredKernel(float time_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float div_x = tex3D(tex_div, coord.x, coord.y - 0.5f, coord.z - 0.5f);
    float coef_x = fminf(0.0f, -div_x * time_step);

    float vort_x = tex3D(tex_vort_x, coord.x, coord.y, coord.z);
    ushort result_x = __float2half_rn(vort_x * __expf(coef_x));
    surf3Dwrite(result_x, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);

    float div_y = tex3D(tex_div, coord.x - 0.5f, coord.y, coord.z - 0.5f);
    float coef_y = fminf(0.0f, -div_y * time_step);

    float vort_y = tex3D(tex_vort_y, coord.x, coord.y, coord.z);
    ushort result_y = __float2half_rn(vort_y * __expf(coef_y));
    surf3Dwrite(result_y, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);

    float div_z = tex3D(tex_div, coord.x - 0.5f, coord.y - 0.5f, coord.z);
    float coef_z = fminf(0.0f, -div_z * time_step);

    float vort_z = tex3D(tex_vort_z, coord.x, coord.y, coord.z);
    ushort result_z = __float2half_rn(vort_z * __expf(coef_z));
    surf3Dwrite(result_z, surf, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void StretchVortexStaggeredKernel(float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float ¦Ø_xx = tex3D(tex_vort_x, coord.x, coord.y, coord.z);
    float ¦Ø_xy = tex3D(tex_vort_y, coord.x + 0.5f, coord.y - 0.5f, coord.z);
    float ¦Ø_xz = tex3D(tex_vort_z, coord.x + 0.5f, coord.y, coord.z - 0.5f);

    float mag_x = sqrtf(¦Ø_xx * ¦Ø_xx + ¦Ø_xy * ¦Ø_xy + ¦Ø_xz * ¦Ø_xz + 0.00001f);
    float dx_x = ¦Ø_xx / mag_x;
    float dy_x = ¦Ø_xy / mag_x;
    float dz_x = ¦Ø_xz / mag_x;

    float v_x0 = tex3D(tex_velocity, coord.x + dx_x + 0.5f, coord.y + dy_x - 0.5f, coord.z + dz_x - 0.5f).x;
    float v_x1 = tex3D(tex_velocity, coord.x - dx_x + 0.5f, coord.y - dy_x - 0.5f, coord.z - dz_x - 0.5f).x;

    ushort result_x = __float2half_rn(scale * (v_x0 - v_x1) * mag_x + ¦Ø_xx);
    surf3Dwrite(result_x, surf_x, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);

    float ¦Ø_yx = tex3D(tex_vort_x, coord.x - 0.5f, coord.y + 0.5f, coord.z);
    float ¦Ø_yy = tex3D(tex_vort_y, coord.x, coord.y, coord.z);
    float ¦Ø_yz = tex3D(tex_vort_z, coord.x, coord.y + 0.5f, coord.z - 0.5f);

    float mag_y = sqrtf(¦Ø_yx * ¦Ø_yx + ¦Ø_yy * ¦Ø_yy + ¦Ø_yz * ¦Ø_yz + 0.00001f);
    float dx_y = ¦Ø_yx / mag_y;
    float dy_y = ¦Ø_yy / mag_y;
    float dz_y = ¦Ø_yz / mag_y;

    float v_y0 = tex3D(tex_velocity, coord.x + dx_y - 0.5f, coord.y + dy_y + 0.5f, coord.z + dz_y - 0.5f).y;
    float v_y1 = tex3D(tex_velocity, coord.x - dx_y - 0.5f, coord.y - dy_y + 0.5f, coord.z - dz_y - 0.5f).y;

    ushort result_y = __float2half_rn(scale * (v_y0 - v_y1) * mag_y + ¦Ø_yy);
    surf3Dwrite(result_y, surf_y, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);

    float ¦Ø_zx = tex3D(tex_vort_x, coord.x - 0.5f, coord.y, coord.z + 0.5f);
    float ¦Ø_zy = tex3D(tex_vort_y, coord.x, coord.y - 0.5f, coord.z + 0.5f);
    float ¦Ø_zz = tex3D(tex_vort_z, coord.x, coord.y, coord.z);

    float mag_z = sqrtf(¦Ø_zx * ¦Ø_zx + ¦Ø_zy * ¦Ø_zy + ¦Ø_zz * ¦Ø_zz + 0.00001f);
    float dx_z = ¦Ø_zx / mag_z;
    float dy_z = ¦Ø_zy / mag_z;
    float dz_z = ¦Ø_zz / mag_z;

    float v_z0 = tex3D(tex_velocity, coord.x + dx_z - 0.5f, coord.y + dy_z - 0.5f, coord.z + dz_z + 0.5f).z;
    float v_z1 = tex3D(tex_velocity, coord.x - dx_z - 0.5f, coord.y - dy_z - 0.5f, coord.z - dz_z + 0.5f).z;

    ushort result_z = __float2half_rn(scale * (v_z0 - v_z1) * mag_z + ¦Ø_zz);
    surf3Dwrite(result_z, surf_z, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchApplyVorticityConfinementStaggered(cudaArray* dest,
                                              cudaArray* velocity,
                                              cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              uint3 volume_size,
                                              BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&tex_velocity, velocity, false,
                                      cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_conf_x = BindHelper::Bind(&tex_conf_x, conf_x, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_conf_x.error() != cudaSuccess)
        return;

    auto bound_conf_y = BindHelper::Bind(&tex_conf_y, conf_y, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_conf_y.error() != cudaSuccess)
        return;

    auto bound_conf_z = BindHelper::Bind(&tex_conf_z, conf_z, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_conf_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ApplyVorticityConfinementStaggeredKernel<<<grid, block>>>();
}

void LaunchBuildVorticityConfinementStaggered(cudaArray* dest_x,
                                              cudaArray* dest_y,
                                              cudaArray* dest_z,
                                              cudaArray* curl_x,
                                              cudaArray* curl_y,
                                              cudaArray* curl_z,
                                              float coeff, float cell_size,
                                              uint3 volume_size,
                                              BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, dest_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, dest_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, dest_z) != cudaSuccess)
        return;

    auto bound_curl_x = BindHelper::Bind(&tex_vort_x, curl_x, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_x.error() != cudaSuccess)
        return;

    auto bound_curl_y = BindHelper::Bind(&tex_vort_y, curl_y, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_y.error() != cudaSuccess)
        return;

    auto bound_curl_z = BindHelper::Bind(&tex_vort_z, curl_z, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    BuildVorticityConfinementStaggeredKernel<<<grid, block>>>(coeff, cell_size,
                                                              0.5f / cell_size,
                                                              volume_size);
}


void LaunchComputeDivergenceStaggeredForVort(cudaArray* dest,
                                             cudaArray* velocity,
                                             float cell_size, uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&tex_velocity, velocity, false,
                                      cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_vel.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeDivergenceStaggeredKernelForVort<<<grid, block>>>(1.0f / cell_size,
                                                             volume_size);
}


void LaunchComputeCurlStaggered(cudaArray* dest_x, cudaArray* dest_y,
                                cudaArray* dest_z, cudaArray* velocity,
                                cudaArray* curl_x, cudaArray* curl_y,
                                cudaArray* curl_z, float inverse_cell_size,
                                uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, dest_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, dest_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, dest_z) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&tex_velocity, velocity, false,
                                      cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_curl_x = BindHelper::Bind(&tex_vort_x, curl_x, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_x.error() != cudaSuccess)
        return;

    auto bound_curl_y = BindHelper::Bind(&tex_vort_y, curl_y, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_y.error() != cudaSuccess)
        return;

    auto bound_curl_z = BindHelper::Bind(&tex_vort_z, curl_z, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_curl_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeCurlStaggeredKernel<<<grid, block>>>(volume_size, inverse_cell_size);
}
