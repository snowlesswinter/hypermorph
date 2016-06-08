#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_b;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_t;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_d;

__global__ void ApplyBuoyancyKernel(float time_step, float ambient_temperature,
                                    float accel_factor, float gravity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float t = tex3D(tex_t, coord.x, coord.y, coord.z);
    float d = tex3D(tex_d, coord.x, coord.y, coord.z);
    float accel =
        time_step * ((t - ambient_temperature) * accel_factor - d * gravity);

    float velocity = tex3D(tex, coord.x, coord.y, coord.z);
    auto result = __float2half_rn(velocity + accel);
    surf3Dwrite(result, surf, x * sizeof(result), y, z, cudaBoundaryModeTrap);
}

__global__ void ApplyBuoyancyStaggeredKernel(float time_step,
                                             float ambient_temperature,
                                             float accel_factor, float gravity,
                                             uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, 0, z) + 0.5f;
    float t_prev = tex3D(tex_t, coord.x, coord.y, coord.z);
    float d_prev = tex3D(tex_d, coord.x, coord.y, coord.z);
    float accel_prev = time_step *
        ((t_prev - ambient_temperature) * accel_factor - d_prev * gravity);

    float y = 1.5f;
    for (int i = 1; i < volume_size.y; i++, y += 1.0f) {
        float t = tex3D(tex_t, coord.x, y, coord.z);
        float d = tex3D(tex_d, coord.x, y, coord.z);
        float accel = time_step *
            ((t - ambient_temperature) * accel_factor - d * gravity);

        float velocity = tex3D(tex, coord.x, y, coord.z);

        auto r = __float2half_rn(velocity + (accel_prev + accel) * 0.5f);
        surf3Dwrite(r, surf, x * sizeof(r), i, z, cudaBoundaryModeTrap);

        t_prev = t;
        d_prev = d;
        accel_prev = accel;
    }
}

__global__ void ComputeDivergenceKernel(float half_inverse_cell_size,
                                        uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float west =     tex3D(tex_x, coord.x - 1.0f, coord.y,        coord.z);
    float center_x = tex3D(tex_x, coord.x,        coord.y,        coord.z);
    float east =     tex3D(tex_x, coord.x + 1.0f, coord.y,        coord.z);
    float south =    tex3D(tex_y, coord.x,        coord.y - 1.0f, coord.z);
    float center_y = tex3D(tex_y, coord.x,        coord.y,        coord.z);
    float north =    tex3D(tex_y, coord.x,        coord.y + 1.0f, coord.z);
    float near =     tex3D(tex_z, coord.x,        coord.y,        coord.z - 1.0f);
    float center_z = tex3D(tex_z, coord.x,        coord.y,        coord.z);
    float far =      tex3D(tex_z, coord.x,        coord.y,        coord.z + 1.0f);

    float diff_ew = east - west;
    float diff_ns = north - south;
    float diff_fn = far - near;

    // Handle boundary problem.
    if (x >= volume_size.x - 1)
        diff_ew = -center_x - west;

    if (x <= 0)
        diff_ew = east + center_x;

    if (y >= volume_size.y - 1)
        diff_ns = -center_y - south;

    if (y <= 0)
        diff_ns = north + center_y;

    if (z >= volume_size.z - 1)
        diff_fn = -center_z - near;

    if (z <= 0)
        diff_fn = far + center_z;

    float div = half_inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    auto r = __float2half_rn(div);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeDivergenceStaggeredKernel(float inverse_cell_size,
                                                 uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float base_x = tex3D(tex_x, coord.x,        coord.y,        coord.z);
    float base_y = tex3D(tex_y, coord.x,        coord.y,        coord.z);
    float base_z = tex3D(tex_z, coord.x,        coord.y,        coord.z);
    float east =   tex3D(tex_x, coord.x + 1.0f, coord.y,        coord.z);
    float north =  tex3D(tex_y, coord.x,        coord.y + 1.0f, coord.z);
    float far =    tex3D(tex_z, coord.x,        coord.y,        coord.z + 1.0f);

    float diff_ew = east  - base_x;
    float diff_ns = north - base_y;
    float diff_fn = far   - base_z;

    // Handle boundary problem
    if (x >= volume_size.x - 1)
        diff_ew = -base_x;

    if (y >= volume_size.y - 1)
        diff_ns = -base_y;

    if (z >= volume_size.z - 1)
        diff_fn = -base_z;

    float div = inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    auto r = __float2half_rn(div);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeResidualDiagnosisKernel(float inverse_h_square,
                                               uint3 volume_size)
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
    float b =      tex3D(tex_b, coord.x, coord.y, coord.z);

    if (coord.y == volume_size.y - 1)
        north = center;

    if (coord.y == 0)
        south = center;

    if (coord.x == volume_size.x - 1)
        east = center;

    if (coord.x == 0)
        west = center;

    if (coord.z == volume_size.z - 1)
        far = center;

    if (coord.z == 0)
        near = center;

    float v = b -
        (north + south + east + west + far + near - 6.0 * center) *
        inverse_h_square;
    surf3Dwrite(fabsf(v), surf, x * sizeof(float), y, z, cudaBoundaryModeTrap);
}

__global__ void RoundPassedKernel(int* dest_array, int round, int x)
{
    dest_array[0] = x * x - round * round;
}

__global__ void SubtractGradientKernel(float half_inverse_cell_size,
                                       uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float near =   tex3D(tex, coord.x, coord.y, coord.z - 1.0f);
    float south =  tex3D(tex, coord.x, coord.y - 1.0f, coord.z);
    float west =   tex3D(tex, coord.x - 1.0f, coord.y, coord.z);
    float center = tex3D(tex, coord.x, coord.y, coord.z);
    float east =   tex3D(tex, coord.x + 1.0f, coord.y, coord.z);
    float north =  tex3D(tex, coord.x, coord.y + 1.0f, coord.z);
    float far =    tex3D(tex, coord.x, coord.y, coord.z + 1.0f);

    float diff_ew = east - west;
    float diff_ns = north - south;
    float diff_fn = far - near;

    // Handle boundary problem
    float3 mask = make_float3(1.0f);
    if (x >= volume_size.x - 1)
        mask.x = 0.0f;

    if (x <= 0)
        mask.x = 0.0f;

    if (y >= volume_size.y - 1)
        mask.y = 0.0f;

    if (y <= 0)
        mask.y = 0.0f;

    if (z >= volume_size.z - 1)
        mask.z = 0.0f;

    if (z <= 0)
        mask.z = 0.0f;

    float old_x = tex3D(tex_x, coord.x, coord.y, coord.z);
    float grad_x = diff_ew * half_inverse_cell_size;
    float new_x = old_x - grad_x;
    auto r_x = __float2half_rn(new_x * mask.x);
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float old_y = tex3D(tex_y, coord.x, coord.y, coord.z);
    float grad_y = diff_ns * half_inverse_cell_size;
    float new_y = old_y - grad_y;
    auto r_y = __float2half_rn(new_y * mask.y);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float old_z = tex3D(tex_z, coord.x, coord.y, coord.z);
    float grad_z = diff_fn * half_inverse_cell_size;
    float new_z = old_z - grad_z;
    auto r_z = __float2half_rn(new_z * mask.z);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

__global__ void SubtractGradientStaggeredKernel(float inverse_cell_size,
                                                uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    
    float near =  tex3D(tex, coord.x,        coord.y,          coord.z - 1.0f);
    float south = tex3D(tex, coord.x,        coord.y - 1.0f,   coord.z);
    float west =  tex3D(tex, coord.x - 1.0f, coord.y,          coord.z);
    float base =  tex3D(tex, coord.x,        coord.y,          coord.z);

    // Handle boundary problem.
    float mask = 1.0f;
    if (x <= 0)
        mask = 0;

    if (y <= 0)
        mask = 0;

    if (z <= 0)
        mask = 0;

    float old_x = tex3D(tex_x, coord.x, coord.y, coord.z);
    float grad_x = (base - west) * inverse_cell_size;
    float new_x = old_x - grad_x;
    auto r_x = __float2half_rn(new_x * mask);
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float old_y = tex3D(tex_y, coord.x, coord.y, coord.z);
    float grad_y = (base - south) * inverse_cell_size;
    float new_y = old_y - grad_y;
    auto r_y = __float2half_rn(new_y * mask);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float old_z = tex3D(tex_z, coord.x, coord.y, coord.z);
    float grad_z = (base - near) * inverse_cell_size;
    float new_z = old_z - grad_z;
    auto r_z = __float2half_rn(new_z * mask);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchApplyBuoyancy(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                         cudaArray* temperature, cudaArray* density,
                         float time_step, float ambient_temperature,
                         float accel_factor, float gravity, uint3 volume_size)
{

    if (BindCudaSurfaceToArray(&surf, vel_y) != cudaSuccess)
        return;

    auto bound_v = BindHelper::Bind(&tex, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_v.error() != cudaSuccess)
        return;

    auto bound_temp = BindHelper::Bind(&tex_t, temperature,
                                       false, cudaFilterModeLinear,
                                       cudaAddressModeClamp);
    if (bound_temp.error() != cudaSuccess)
        return;

    auto bound_density = BindHelper::Bind(&tex_d, density,
                                          false, cudaFilterModeLinear,
                                          cudaAddressModeClamp);
    if (bound_density.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ApplyBuoyancyKernel<<<grid, block>>>(time_step, ambient_temperature,
                                         accel_factor, gravity);
}

void LaunchApplyBuoyancyStaggered(cudaArray* vel_x, cudaArray* vel_y,
                                  cudaArray* vel_z, cudaArray* temperature,
                                  cudaArray* density, float time_step,
                                  float ambient_temperature, float accel_factor,
                                  float gravity, uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, vel_y) != cudaSuccess)
        return;

    auto bound_v = BindHelper::Bind(&tex, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_v.error() != cudaSuccess)
        return;

    auto bound_t = BindHelper::Bind(&tex_t, temperature, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_t.error() != cudaSuccess)
        return;

    auto bound_d = BindHelper::Bind(&tex_d, density, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_d.error() != cudaSuccess)
        return;

    dim3 block(16, 1, 16);
    dim3 grid(volume_size.x / block.x, 1, volume_size.z / block.z);
    ApplyBuoyancyStaggeredKernel<<<grid, block>>>(time_step,
                                                  ambient_temperature,
                                                  accel_factor, gravity,
                                                  volume_size);
}

void LaunchComputeDivergence(cudaArray* div, cudaArray* vel_x, cudaArray* vel_y,
                             cudaArray* vel_z, float half_inverse_cell_size,
                             uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, div) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vel_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vel_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeDivergenceKernel<<<grid, block>>>(half_inverse_cell_size,
                                             volume_size);
}

void LaunchComputeDivergenceStaggered(cudaArray* div, cudaArray* vel_x,
                                      cudaArray* vel_y, cudaArray* vel_z,
                                      float inverse_cell_size,
                                      uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, div) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vel_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vel_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeDivergenceStaggeredKernel<<<grid, block>>>(inverse_cell_size,
                                                      volume_size);
}

void LaunchComputeResidualDiagnosis(cudaArray* residual, cudaArray* u,
                                    cudaArray* b, float inverse_h_square,
                                    uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&surf, residual) != cudaSuccess)
        return;

    auto bound_u = BindHelper::Bind(&tex, u, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_u.error() != cudaSuccess)
        return;

    auto bound_b = BindHelper::Bind(&tex_b, b, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_b.error() != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeResidualDiagnosisKernel<<<grid, block>>>(inverse_h_square,
                                                    volume_size);
}

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
}

void LaunchSubtractGradient(cudaArray* vel_x, cudaArray* vel_y,
                            cudaArray* vel_z, cudaArray* pressure,
                            float half_inverse_cell_size, uint3 volume_size,
                            BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vel_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vel_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vel_z) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vel_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vel_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, pressure, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    SubtractGradientKernel<<<grid, block>>>(half_inverse_cell_size,
                                            volume_size);
}

void LaunchSubtractGradientStaggered(cudaArray* vel_x, cudaArray* vel_y,
                                     cudaArray* vel_z, cudaArray* pressure,
                                     float inverse_cell_size, uint3 volume_size,
                                     BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vel_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vel_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vel_z) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vel_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vel_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, pressure, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    SubtractGradientStaggeredKernel<<<grid, block>>>(inverse_cell_size,
                                                     volume_size);
}
