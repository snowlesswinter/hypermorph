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
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_xp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_yp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_zp;

__global__ void AddCurlPsiKernel(float inverse_cell_size, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Keep away from the shearing boundaries.
    int d = 4;
    if (x <= d || y <= d || z <= d || x >= volume_size.x - d ||
            y >= volume_size.y - d || z >= volume_size.z - d)
        return;

    float3 coord = make_float3(x, y, z);

    float ��_y0 = tex3D(tex_y, coord.x, coord.y, coord.z);
    float ��_y1 = tex3D(tex_y, coord.x, coord.y, coord.z + 1.0f);
    float ��_z0 = tex3D(tex_z, coord.x, coord.y, coord.z);
    float ��_z1 = tex3D(tex_z, coord.x, coord.y + 1.0f, coord.z);

    float u = inverse_cell_size * (��_z1 - ��_z0 - ��_y1 + ��_y0);

    float ��_x0 = tex3D(tex_x, coord.x, coord.y, coord.z);
    float ��_x1 = tex3D(tex_x, coord.x, coord.y, coord.z + 1.0f);
    float ��_z2 = tex3D(tex_z, coord.x + 1.0f, coord.y, coord.z);

    float v = inverse_cell_size * (��_x1 - ��_x0 - ��_z2 + ��_z0);

    float ��_y2 = tex3D(tex_y, coord.x + 1.0f, coord.y, coord.z);
    float ��_x2 = tex3D(tex_x, coord.x, coord.y + 1.0f, coord.z);
    float w = inverse_cell_size * (��_y2 - ��_y0 - ��_x2 + ��_x0);

    auto result = make_ushort4(__float2half_rn(u),
                               __float2half_rn(v),
                               __float2half_rn(w),
                               0);
    surf3Dwrite(result, surf, x * sizeof(result), y, z, cudaBoundaryModeTrap);
}

__global__ void ApplyVorticityConfinementStaggeredKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x == 0 || y == 0 || z == 0)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float conf_x = tex3D(tex_x, coord.x - 0.5f, coord.y + 0.5f, coord.z + 0.5f);
    float conf_y = tex3D(tex_y, coord.x + 0.5f, coord.y - 0.5f, coord.z + 0.5f);
    float conf_z = tex3D(tex_z, coord.x + 0.5f, coord.y + 0.5f, coord.z - 0.5f);
    float4 velocity = tex3D(tex_velocity, coord.x, coord.y, coord.z);
    auto result = make_ushort4(__float2half_rn(velocity.x + conf_x),
                               __float2half_rn(velocity.y + conf_y),
                               __float2half_rn(velocity.z + conf_z),
                               0);
    surf3Dwrite(result, surf, x * sizeof(result), y, z, cudaBoundaryModeTrap);
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
    float near_vort_x =   tex3D(tex_x, coord.x,        coord.y + 0.5f, coord.z - 0.5f);
    float near_vort_y =   tex3D(tex_y, coord.x + 0.5f, coord.y,        coord.z - 0.5f);
    float near_vort_z =   tex3D(tex_z, coord.x + 0.5f, coord.y + 0.5f, coord.z - 1.0f);
    float near_vort = sqrtf(near_vort_x * near_vort_x + near_vort_y * near_vort_y + near_vort_z * near_vort_z);

    float south_vort_x =  tex3D(tex_x, coord.x,        coord.y - 0.5f, coord.z + 0.5f);
    float south_vort_y =  tex3D(tex_y, coord.x + 0.5f, coord.y - 1.0f, coord.z + 0.5f);
    float south_vort_z =  tex3D(tex_z, coord.x + 0.5f, coord.y - 0.5f, coord.z);
    float south_vort = sqrtf(south_vort_x * south_vort_x + south_vort_y * south_vort_y + south_vort_z * south_vort_z);

    float west_vort_x =   tex3D(tex_x, coord.x - 1.0f, coord.y + 0.5f, coord.z + 0.5f);
    float west_vort_y =   tex3D(tex_y, coord.x - 0.5f, coord.y,        coord.z + 0.5f);
    float west_vort_z =   tex3D(tex_z, coord.x - 0.5f, coord.y + 0.5f, coord.z);
    float west_vort = sqrtf(west_vort_x * west_vort_x + west_vort_y * west_vort_y + west_vort_z * west_vort_z);

    float center_vort_x = tex3D(tex_x, coord.x,        coord.y,        coord.z);
    float center_vort_y = tex3D(tex_y, coord.x,        coord.y,        coord.z);
    float center_vort_z = tex3D(tex_z, coord.x,        coord.y,        coord.z);

    float east_vort_x =   tex3D(tex_x, coord.x + 1.0f, coord.y + 0.5f, coord.z + 0.5f);
    float east_vort_y =   tex3D(tex_y, coord.x + 1.5f, coord.y,        coord.z + 0.5f);
    float east_vort_z =   tex3D(tex_z, coord.x + 1.5f, coord.y + 0.5f, coord.z);
    float east_vort = sqrtf(east_vort_x * east_vort_x + east_vort_y * east_vort_y + east_vort_z * east_vort_z);

    float north_vort_x =  tex3D(tex_x, coord.x,        coord.y + 1.5f, coord.z + 0.5f);
    float north_vort_y =  tex3D(tex_y, coord.x + 0.5f, coord.y + 1.0f, coord.z + 0.5f);
    float north_vort_z =  tex3D(tex_z, coord.x + 0.5f, coord.y + 1.5f, coord.z);
    float north_vort = sqrtf(north_vort_x * north_vort_x + north_vort_y * north_vort_y + north_vort_z * north_vort_z);

    float far_vort_x =    tex3D(tex_x, coord.x,        coord.y + 0.5f, coord.z + 1.5f);
    float far_vort_y =    tex3D(tex_y, coord.x + 0.5f, coord.y,        coord.z + 1.5f);
    float far_vort_z =    tex3D(tex_z, coord.x + 0.5f, coord.y + 0.5f, coord.z + 1.0f);
    float far_vort = sqrtf(far_vort_x * far_vort_x + far_vort_y * far_vort_y + far_vort_z * far_vort_z);

    // Calculate normalized ��.
    float ��_x = half_inverse_cell_size * (east_vort - west_vort);
    float ��_y = half_inverse_cell_size * (north_vort - south_vort);
    float ��_z = half_inverse_cell_size * (far_vort - near_vort);

    float ��_mag = sqrtf(��_x * ��_x + ��_y * ��_y + ��_z * ��_z + 0.00001f);
    ��_x /= ��_mag;
    ��_y /= ��_mag;
    ��_z /= ��_mag;

    // Vorticity confinement at the center of the grid.
    float conf_x = coeff * cell_size * (��_y * center_vort_z - ��_z * center_vort_y);
    float conf_y = coeff * cell_size * (��_z * center_vort_x - ��_x * center_vort_z);
    float conf_z = coeff * cell_size * (��_x * center_vort_y - ��_y * center_vort_x);

    ushort result_x = __float2half_rn(conf_x);
    surf3Dwrite(result_x, surf_x, x * sizeof(result_x), y, z, cudaBoundaryModeTrap);

    ushort result_y = __float2half_rn(conf_y);
    surf3Dwrite(result_y, surf_y, x * sizeof(result_y), y, z, cudaBoundaryModeTrap);

    ushort result_z = __float2half_rn(conf_z);
    surf3Dwrite(result_z, surf_z, x * sizeof(result_z), y, z, cudaBoundaryModeTrap);
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
            result.x = tex3D(tex_x, coord.x + 1.0f, coord.y, coord.z);
            result.y = tex3D(tex_y, coord.x + 1.0f, coord.y, coord.z);
            result.z = tex3D(tex_z, coord.x + 1.0f, coord.y, coord.z);
        } else if (y == 0) {
            result.x = tex3D(tex_x, coord.x, coord.y + 1.0f, coord.z);
            result.y = tex3D(tex_y, coord.x, coord.y + 1.0f, coord.z);
            result.z = tex3D(tex_z, coord.x, coord.y + 1.0f, coord.z);
        } else {
            result.x = tex3D(tex_x, coord.x, coord.y, coord.z + 1.0f);
            result.y = tex3D(tex_y, coord.x, coord.y, coord.z + 1.0f);
            result.z = tex3D(tex_z, coord.x, coord.y, coord.z + 1.0f);
        }
    } else if (x == volume_size.x - 1) {
        result.x = tex3D(tex_x, coord.x - 1.0f, coord.y, coord.z);
        result.y = tex3D(tex_y, coord.x - 1.0f, coord.y, coord.z);
        result.z = tex3D(tex_z, coord.x - 1.0f, coord.y, coord.z);
    } else if (y == volume_size.y - 1) {
        result.x = tex3D(tex_x, coord.x, coord.y - 1.0f, coord.z);
        result.y = tex3D(tex_y, coord.x, coord.y - 1.0f, coord.z);
        result.z = tex3D(tex_z, coord.x, coord.y - 1.0f, coord.z);
    } else {
        result.x = tex3D(tex_x, coord.x, coord.y, coord.z - 1.0f);
        result.y = tex3D(tex_y, coord.x, coord.y, coord.z - 1.0f);
        result.z = tex3D(tex_z, coord.x, coord.y, coord.z - 1.0f);
    }

    auto raw_x = __float2half_rn(result.x);
    surf3Dwrite(raw_x, surf_x, x * sizeof(raw_x), y, z, cudaBoundaryModeTrap);

    auto raw_y = __float2half_rn(result.y);
    surf3Dwrite(raw_y, surf_y, x * sizeof(raw_y), y, z, cudaBoundaryModeTrap);

    auto raw_z = __float2half_rn(result.z);
    surf3Dwrite(raw_z, surf_z, x * sizeof(raw_z), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeDeltaVorticityKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float v_x =  tex3D(tex_x,  coord.x, coord.y, coord.z);
    float v_xp = tex3D(tex_xp, coord.x, coord.y, coord.z);
    auto raw_x = __float2half_rn(v_xp - v_x);
    surf3Dwrite(raw_x, surf_x, x * sizeof(raw_x), y, z, cudaBoundaryModeTrap);

    float v_y =  tex3D(tex_y,  coord.x, coord.y, coord.z);
    float v_yp = tex3D(tex_yp, coord.x, coord.y, coord.z);
    auto raw_y = __float2half_rn(v_yp - v_y);
    surf3Dwrite(raw_y, surf_y, x * sizeof(raw_y), y, z, cudaBoundaryModeTrap);
    
    float v_z =  tex3D(tex_z,  coord.x, coord.y, coord.z);
    float v_zp = tex3D(tex_zp, coord.x, coord.y, coord.z);
    auto raw_z = __float2half_rn(v_zp - v_z);
    surf3Dwrite(raw_z, surf_z, x * sizeof(raw_z), y, z, cudaBoundaryModeTrap);
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
    auto result = __float2half_rn(div);
    surf3Dwrite(result, surf, x * sizeof(result), y, z, cudaBoundaryModeTrap);
}

__global__ void DecayVorticesStaggeredKernel(float time_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float div_x = tex3D(tex, coord.x, coord.y - 0.5f, coord.z - 0.5f);
    float coef_x = fminf(0.0f, -div_x * time_step);

    float vort_x = tex3D(tex_x, coord.x, coord.y, coord.z);
    auto r_x = __float2half_rn(vort_x * __expf(coef_x));
    surf3Dwrite(r_x, surf, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float div_y = tex3D(tex, coord.x - 0.5f, coord.y, coord.z - 0.5f);
    float coef_y = fminf(0.0f, -div_y * time_step);

    float vort_y = tex3D(tex_y, coord.x, coord.y, coord.z);
    auto r_y = __float2half_rn(vort_y * __expf(coef_y));
    surf3Dwrite(r_y, surf, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float div_z = tex3D(tex, coord.x - 0.5f, coord.y - 0.5f, coord.z);
    float coef_z = fminf(0.0f, -div_z * time_step);

    float vort_z = tex3D(tex_z, coord.x, coord.y, coord.z);
    auto r_z = __float2half_rn(vort_z * __expf(coef_z));
    surf3Dwrite(r_z, surf, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

__global__ void StretchVorticesStaggeredKernel(float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float ��_xx = tex3D(tex_x, coord.x, coord.y, coord.z);
    float ��_xy = tex3D(tex_y, coord.x + 0.5f, coord.y - 0.5f, coord.z);
    float ��_xz = tex3D(tex_z, coord.x + 0.5f, coord.y, coord.z - 0.5f);

    float mag_x = sqrtf(��_xx * ��_xx + ��_xy * ��_xy + ��_xz * ��_xz + 0.00001f);
    float dx_x = ��_xx / mag_x;
    float dy_x = ��_xy / mag_x;
    float dz_x = ��_xz / mag_x;

    float v_x0 = tex3D(tex_velocity, coord.x + dx_x + 0.5f, coord.y + dy_x - 0.5f, coord.z + dz_x - 0.5f).x;
    float v_x1 = tex3D(tex_velocity, coord.x - dx_x + 0.5f, coord.y - dy_x - 0.5f, coord.z - dz_x - 0.5f).x;

    auto result_x = __float2half_rn(scale * (v_x0 - v_x1) * mag_x + ��_xx);
    surf3Dwrite(result_x, surf_x, x * sizeof(result_x), y, z, cudaBoundaryModeTrap);

    float ��_yx = tex3D(tex_x, coord.x - 0.5f, coord.y + 0.5f, coord.z);
    float ��_yy = tex3D(tex_y, coord.x, coord.y, coord.z);
    float ��_yz = tex3D(tex_z, coord.x, coord.y + 0.5f, coord.z - 0.5f);

    float mag_y = sqrtf(��_yx * ��_yx + ��_yy * ��_yy + ��_yz * ��_yz + 0.00001f);
    float dx_y = ��_yx / mag_y;
    float dy_y = ��_yy / mag_y;
    float dz_y = ��_yz / mag_y;

    float v_y0 = tex3D(tex_velocity, coord.x + dx_y - 0.5f, coord.y + dy_y + 0.5f, coord.z + dz_y - 0.5f).y;
    float v_y1 = tex3D(tex_velocity, coord.x - dx_y - 0.5f, coord.y - dy_y + 0.5f, coord.z - dz_y - 0.5f).y;

    auto result_y = __float2half_rn(scale * (v_y0 - v_y1) * mag_y + ��_yy);
    surf3Dwrite(result_y, surf_y, x * sizeof(result_y), y, z, cudaBoundaryModeTrap);

    float ��_zx = tex3D(tex_x, coord.x - 0.5f, coord.y, coord.z + 0.5f);
    float ��_zy = tex3D(tex_y, coord.x, coord.y - 0.5f, coord.z + 0.5f);
    float ��_zz = tex3D(tex_z, coord.x, coord.y, coord.z);

    float mag_z = sqrtf(��_zx * ��_zx + ��_zy * ��_zy + ��_zz * ��_zz + 0.00001f);
    float dx_z = ��_zx / mag_z;
    float dy_z = ��_zy / mag_z;
    float dz_z = ��_zz / mag_z;

    float v_z0 = tex3D(tex_velocity, coord.x + dx_z - 0.5f, coord.y + dy_z - 0.5f, coord.z + dz_z + 0.5f).z;
    float v_z1 = tex3D(tex_velocity, coord.x - dx_z - 0.5f, coord.y - dy_z - 0.5f, coord.z - dz_z + 0.5f).z;

    auto result_z = __float2half_rn(scale * (v_z0 - v_z1) * mag_z + ��_zz);
    surf3Dwrite(result_z, surf_z, x * sizeof(result_z), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAddCurlPsi(cudaArray* velocity, cudaArray* psi_x, cudaArray* psi_y,
                      cudaArray* psi_z, float cell_size, uint3 volume_size,
                      BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, velocity) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, psi_x, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, psi_y, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, psi_z, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    AddCurlPsiKernel<<<grid, block>>>(1.0f / cell_size, volume_size);
}

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

    auto bound_x = BindHelper::Bind(&tex_x, conf_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, conf_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, conf_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
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

    auto bound_x = BindHelper::Bind(&tex_x, curl_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, curl_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, curl_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    BuildVorticityConfinementStaggeredKernel<<<grid, block>>>(coeff, cell_size,
                                                              0.5f / cell_size,
                                                              volume_size);
}


void LaunchComputeDivergenceStaggeredForVort(cudaArray* div,
                                             cudaArray* velocity,
                                             float cell_size, uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, div) != cudaSuccess)
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

    auto bound_x = BindHelper::Bind(&tex_x, curl_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, curl_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, curl_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeCurlStaggeredKernel<<<grid, block>>>(volume_size, inverse_cell_size);
}

void LaunchComputeDeltaVorticity(cudaArray* vnp1_x, cudaArray* vnp1_y,
                                 cudaArray* vnp1_z,  cudaArray* vn_x,
                                 cudaArray* vn_y, cudaArray* vn_z,
                                 uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vnp1_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vnp1_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vnp1_z) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vn_x, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vn_y, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vn_z, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    auto bound_xp = BindHelper::Bind(&tex_xp, vnp1_x, false,
                                     cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_xp.error() != cudaSuccess)
        return;

    auto bound_yp = BindHelper::Bind(&tex_yp, vnp1_y, false,
                                     cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_yp.error() != cudaSuccess)
        return;

    auto bound_zp = BindHelper::Bind(&tex_zp, vnp1_z, false,
                                     cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_zp.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeDeltaVorticityKernel<<<grid, block>>>();
}

void LaunchDecayVorticesStaggered(cudaArray* vort_x, cudaArray* vort_y,
                                  cudaArray* vort_z, cudaArray* div,
                                  float time_step, uint3 volume_size,
                                  BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vort_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vort_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vort_z) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, div, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    DecayVorticesStaggeredKernel<<<grid, block>>>(time_step);
}

void LaunchStretchVorticesStaggered(cudaArray* vort_np1_x,
                                    cudaArray* vort_np1_y,
                                    cudaArray* vort_np1_z, cudaArray* velocity,
                                    cudaArray* vort_x, cudaArray* vort_y,
                                    cudaArray* vort_z, float cell_size,
                                    float time_step, uint3 volume_size,
                                    BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vort_np1_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vort_np1_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vort_np1_z) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&tex_velocity, velocity, false,
                                      cudaFilterModeLinear,
                                      cudaAddressModeClamp);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vort_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vort_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vort_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    float scale = time_step / (cell_size * 2.0f);

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    StretchVorticesStaggeredKernel<<<grid, block>>>(scale);
}
