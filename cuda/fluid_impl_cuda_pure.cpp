#include "fluid_impl_cuda_pure.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>
#include <driver_types.h>

#include "graphics_resource.h"
#include "../vmath.hpp"

extern void LaunchAdvectPure(cudaArray_t dest_array, cudaArray_t velocity_array,
                             cudaArray_t source_array, float time_step,
                             float dissipation, int3 volume_size);
extern void LaunchAdvectVelocityPure(cudaArray_t dest_array,
                                     cudaArray_t velocity_array,
                                     float time_step, float dissipation,
                                     int3 volume_size);
extern void LaunchApplyBuoyancyPure(cudaArray* dest_array,
                                    cudaArray* velocity_array,
                                    cudaArray* temperature_array,
                                    float time_step, float ambient_temperature,
                                    float accel_factor, float gravity,
                                    int3 volume_size);
extern void LaunchApplyImpulsePure(cudaArray* dest_array,
                                   cudaArray* original_array,
                                   float3 center_point, float3 hotspot,
                                   float radius, float value, int3 volume_size);
extern void LaunchComputeDivergencePure(cudaArray* dest_array,
                                        cudaArray* velocity_array,
                                        float half_inverse_cell_size,
                                        int3 volume_size);
extern void LaunchComputeResidualPackedDiagnosis(cudaArray* dest_array,
                                                 cudaArray* source_array,
                                                 float inverse_h_square,
                                                 int3 volume_size);
extern void LaunchDampedJacobiPure(cudaArray* dest_array,
                                   cudaArray* packed_array,
                                   float one_minus_omega,
                                   float minus_square_cell_size,
                                   float omega_over_beta, int3 volume_size);
extern void LaunchSubstractGradientPure(cudaArray* dest_array,
                                        cudaArray* packed_array,
                                        float gradient_scale, int3 volume_size);

namespace
{
int3 FromVmathVector(const vmath::Vector3& v)
{
    return make_int3(static_cast<int>(v.getX()), static_cast<int>(v.getY()),
                     static_cast<int>(v.getZ()));
}
} // Anonymous namespace.

FluidImplCudaPure::FluidImplCudaPure()
{

}

FluidImplCudaPure::~FluidImplCudaPure()
{
}

void FluidImplCudaPure::Advect(cudaArray* dest, cudaArray* velocity,
                               cudaArray* source, float time_step,
                               float dissipation,
                               const Vectormath::Aos::Vector3& volume_size)
{
    LaunchAdvectPure(dest, velocity, source, time_step, dissipation,
                     FromVmathVector(volume_size));
}

void FluidImplCudaPure::AdvectDensity(GraphicsResource* dest,
                                      cudaArray* velocity,
                                      GraphicsResource* density,
                                      float time_step, float dissipation,
                                      const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        dest->resource(), density->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Destination texture.
    cudaArray* dest_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&dest_array,
                                                   dest->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Source texture.
    cudaArray* source_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&source_array,
                                                   density->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchAdvectPure(dest_array, velocity, source_array, time_step, dissipation,
                     FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCudaPure::AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                                       float time_step, float dissipation,
                                       const vmath::Vector3& volume_size)
{
    LaunchAdvectVelocityPure(dest, velocity, time_step, dissipation,
                             FromVmathVector(volume_size));
}

void FluidImplCudaPure::ApplyBuoyancy(cudaArray* dest, cudaArray* velocity,
                                      cudaArray* temperature, float time_step,
                                      float ambient_temperature,
                                      float accel_factor, float gravity,
                                      const vmath::Vector3& volume_size)
{
    LaunchApplyBuoyancyPure(dest, velocity, temperature, time_step,
                            ambient_temperature, accel_factor, gravity,
                            FromVmathVector(volume_size));
}

void FluidImplCudaPure::ApplyImpulse(cudaArray* dest, cudaArray* source,
                                     const vmath::Vector3& center_point,
                                     const vmath::Vector3& hotspot,
                                     float radius, float value,
                                     const vmath::Vector3& volume_size)
{
    LaunchApplyImpulsePure(
        dest, source,
        make_float3(center_point.getX(), center_point.getY(),
                    center_point.getZ()),
        make_float3(hotspot.getX(), hotspot.getY(), hotspot.getZ()),
        radius, value, FromVmathVector(volume_size));
}

void FluidImplCudaPure::ApplyImpulseDensity(GraphicsResource* dest,
                                            GraphicsResource* density,
                                            const vmath::Vector3& center_point,
                                            const vmath::Vector3& hotspot,
                                            float radius, float value,
                                            const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        dest->resource(), density->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Destination texture.
    cudaArray* dest_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&dest_array,
                                                   dest->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Source texture.
    cudaArray* source_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&source_array,
                                                   density->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchApplyImpulsePure(
        dest_array, source_array,
        make_float3(center_point.getX(), center_point.getY(),
                    center_point.getZ()),
        make_float3(hotspot.getX(), hotspot.getY(), hotspot.getZ()),
        radius, value, FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCudaPure::ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                                          float half_inverse_cell_size,
                                          const vmath::Vector3& volume_size)
{
    LaunchComputeDivergencePure(dest, velocity, half_inverse_cell_size,
                                FromVmathVector(volume_size));
}

void FluidImplCudaPure::ComputeResidualPackedDiagnosis(
    cudaArray* dest, cudaArray* source, float inverse_h_square,
    const vmath::Vector3& volume_size)
{
    LaunchComputeResidualPackedDiagnosis(dest, source, inverse_h_square,
                                         FromVmathVector(volume_size));
}

void FluidImplCudaPure::DampedJacobi(cudaArray* dest, cudaArray* packed,
                                     float one_minus_omega,
                                     float minus_square_cell_size,
                                     float omega_over_beta,
                                     const vmath::Vector3& volume_size)
{
    LaunchDampedJacobiPure(dest, packed, one_minus_omega,
                           minus_square_cell_size, omega_over_beta,
                           FromVmathVector(volume_size));
}

void FluidImplCudaPure::SubstractGradient(cudaArray* dest, cudaArray* packed,
                                          float gradient_scale,
                                          const vmath::Vector3& volume_size)
{
    LaunchSubstractGradientPure(dest, packed, gradient_scale,
                                FromVmathVector(volume_size));
}
