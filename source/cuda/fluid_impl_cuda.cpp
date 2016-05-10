#include "fluid_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>
#include <driver_types.h>

#include "graphics_resource.h"
#include "third_party/glm/vec3.hpp"

extern void LaunchAdvect(cudaArray_t dest_array, cudaArray_t velocity_array,
                         cudaArray_t source_array, float time_step,
                         float dissipation, uint3 volume_size);
extern void LaunchAdvectVelocity(cudaArray_t dest_array,
                                 cudaArray_t velocity_array, float time_step,
                                 float dissipation, uint3 volume_size);
extern void LaunchApplyBuoyancy(cudaArray* dest_array,
                                cudaArray* velocity_array,
                                cudaArray* temperature_array,
                                float time_step, float ambient_temperature,
                                float accel_factor, float gravity,
                                uint3 volume_size);
extern void LaunchApplyImpulse(cudaArray* dest_array, cudaArray* original_array,
                               float3 center_point, float3 hotspot,
                               float radius, float3 value, uint32_t mask,
                               uint3 volume_size);
extern void LaunchComputeDivergence(cudaArray* dest_array,
                                    cudaArray* velocity_array,
                                    float half_inverse_cell_size,
                                    uint3 volume_size);
extern void LaunchComputeResidualPackedDiagnosis(cudaArray* dest_array,
                                                 cudaArray* source_array,
                                                 float inverse_h_square,
                                                 uint3 volume_size);
extern void LaunchDampedJacobi(cudaArray* dest_array, cudaArray* source_array,
                               float minus_square_cell_size,
                               float omega_over_beta, int num_of_iterations,
                               uint3 volume_size, BlockArrangement* ba);
extern void LaunchRoundPassed(int* dest_array, int round, int x);
extern void LaunchSubtractGradient(cudaArray* dest_array,
                                   cudaArray* packed_array,
                                   float gradient_scale, uint3 volume_size,
                                   BlockArrangement* ba);

namespace
{
uint3 FromVmathVector(const glm::ivec3& v)
{
    return make_uint3(static_cast<uint>(v.x), static_cast<uint>(v.y),
                      static_cast<uint>(v.z));
}
} // Anonymous namespace.

FluidImplCudaPure::FluidImplCudaPure(BlockArrangement* ba)
    : ba_(ba)
{

}

FluidImplCudaPure::~FluidImplCudaPure()
{
}

void FluidImplCudaPure::Advect(cudaArray* dest, cudaArray* velocity,
                               cudaArray* source, float time_step,
                               float dissipation,
                               const glm::ivec3& volume_size)
{
    LaunchAdvect(dest, velocity, source, time_step, dissipation,
                 FromVmathVector(volume_size));
}

void FluidImplCudaPure::AdvectDensity(cudaArray* dest, cudaArray* velocity,
                                      cudaArray* density, float time_step,
                                      float dissipation,
                                      const glm::ivec3& volume_size)
{
    LaunchAdvect(dest, velocity, density, time_step, dissipation,
                 FromVmathVector(volume_size));
}

void FluidImplCudaPure::AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                                       float time_step, float dissipation,
                                       const glm::ivec3& volume_size)
{
    LaunchAdvectVelocity(dest, velocity, time_step, dissipation,
                         FromVmathVector(volume_size));
}

void FluidImplCudaPure::ApplyBuoyancy(cudaArray* dest, cudaArray* velocity,
                                      cudaArray* temperature, float time_step,
                                      float ambient_temperature,
                                      float accel_factor, float gravity,
                                      const glm::ivec3& volume_size)
{
    LaunchApplyBuoyancy(dest, velocity, temperature, time_step,
                        ambient_temperature, accel_factor, gravity,
                        FromVmathVector(volume_size));
}

void FluidImplCudaPure::ApplyImpulse(cudaArray* dest, cudaArray* source,
                                     const glm::vec3& center_point,
                                     const glm::vec3& hotspot, float radius,
                                     const glm::vec3& value,
                                     uint32_t mask,
                                     const glm::ivec3& volume_size)
{
    LaunchApplyImpulse(
        dest, source,
        make_float3(center_point.x, center_point.y, center_point.z),
        make_float3(hotspot.x, hotspot.y, hotspot.z),
        radius, make_float3(value.x, value.y, value.z), mask,
        FromVmathVector(volume_size));
}

void FluidImplCudaPure::ApplyImpulseDensity(cudaArray* density,
                                            const glm::vec3& center_point,
                                            const glm::vec3& hotspot,
                                            float radius, float value,
                                            const glm::ivec3& volume_size)
{
    LaunchApplyImpulse(
        density, density,
        make_float3(center_point.x, center_point.y, center_point.z),
        make_float3(hotspot.x, hotspot.y, hotspot.z),
        radius, make_float3(value, 0, 0), 1, FromVmathVector(volume_size));
}

void FluidImplCudaPure::ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                                          float half_inverse_cell_size,
                                          const glm::ivec3& volume_size)
{
    LaunchComputeDivergence(dest, velocity, half_inverse_cell_size,
                            FromVmathVector(volume_size));
}

void FluidImplCudaPure::ComputeResidualPackedDiagnosis(
    cudaArray* dest, cudaArray* source, float inverse_h_square,
    const glm::ivec3& volume_size)
{
    LaunchComputeResidualPackedDiagnosis(dest, source, inverse_h_square,
                                         FromVmathVector(volume_size));
}

void FluidImplCudaPure::DampedJacobi(cudaArray* dest, cudaArray* source,
                                     float minus_square_cell_size,
                                     float omega_over_beta,
                                     int num_of_iterations,
                                     const glm::ivec3& volume_size)
{
    LaunchDampedJacobi(dest, source, minus_square_cell_size, omega_over_beta,
                       num_of_iterations, FromVmathVector(volume_size), ba_);
}

void FluidImplCudaPure::SubtractGradient(cudaArray* dest, cudaArray* packed,
                                         float gradient_scale,
                                         const glm::ivec3& volume_size)
{
    LaunchSubtractGradient(dest, packed, gradient_scale,
                           FromVmathVector(volume_size), ba_);
}

void FluidImplCudaPure::RoundPassed(int round)
{
    int* dest_array = nullptr;
    cudaError_t result = cudaMalloc(&dest_array, 4);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchRoundPassed(dest_array, round, 3);

    cudaFree(dest_array);
}