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

extern void LaunchAdvectScalar(cudaArray_t dest_array,
                               cudaArray_t velocity_array,
                               cudaArray_t source_array,
                               cudaArray_t intermediate_array, float time_step,
                               float dissipation, bool quadratic_dissipation,
                               uint3 volume_size, AdvectionMethod method);
extern void LaunchAdvectScalarStaggered(cudaArray_t dest_array,
                                        cudaArray_t velocity_array,
                                        cudaArray_t source_array,
                                        cudaArray_t intermediate_array,
                                        float time_step, float dissipation,
                                        bool quadratic_dissipation,
                                        uint3 volume_size,
                                        AdvectionMethod method);
extern void LaunchAdvectVelocity(cudaArray_t dest_array,
                                 cudaArray_t velocity_array,
                                 cudaArray_t intermediate_array,
                                 float time_step, float time_step_prev,
                                 float dissipation, uint3 volume_size,
                                 AdvectionMethod method);
extern void LaunchAdvectVelocityStaggered(cudaArray_t dest_array,
                                          cudaArray_t velocity_array,
                                          cudaArray_t intermediate_array,
                                          float time_step, float time_step_prev,
                                          float dissipation, uint3 volume_size,
                                          AdvectionMethod method);
extern void LaunchApplyBuoyancy(cudaArray* dest_array,
                                cudaArray* velocity_array,
                                cudaArray* temperature_array,
                                float time_step, float ambient_temperature,
                                float accel_factor, float gravity,
                                uint3 volume_size);
extern void LaunchApplyBuoyancyStaggered(cudaArray* dest_array,
                                         cudaArray* velocity_array,
                                         cudaArray* temperature_array,
                                         float time_step,
                                         float ambient_temperature,
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
extern void LaunchComputeDivergenceStaggered(cudaArray* dest_array,
                                             cudaArray* velocity_array,
                                             float inverse_cell_size,
                                             uint3 volume_size);
extern void LaunchComputeResidualPackedDiagnosis(cudaArray* dest_array,
                                                 cudaArray* source_array,
                                                 float inverse_h_square,
                                                 uint3 volume_size);
extern void LaunchDampedJacobi(cudaArray* dest_array, cudaArray* source_array,
                               float minus_square_cell_size,
                               float omega_over_beta, int num_of_iterations,
                               uint3 volume_size, BlockArrangement* ba);
extern void LaunchImpulseDensity(cudaArray* dest_array,
                                 cudaArray* original_array, float3 center_point,
                                 float radius, float3 value, uint3 volume_size);
extern void LaunchRoundPassed(int* dest_array, int round, int x);
extern void LaunchSubtractGradient(cudaArray* dest_array,
                                   cudaArray* packed_array,
                                   float half_inverse_cell_size,
                                   uint3 volume_size, BlockArrangement* ba);
extern void LaunchSubtractGradientStaggered(cudaArray* dest_array,
                                            cudaArray* packed_array,
                                            float inverse_cell_size,
                                            uint3 volume_size,
                                            BlockArrangement* ba);

namespace
{
uint3 FromGlmVector(const glm::ivec3& v)
{
    return make_uint3(static_cast<uint>(v.x), static_cast<uint>(v.y),
                      static_cast<uint>(v.z));
}
} // Anonymous namespace.

FluidImplCuda::FluidImplCuda(BlockArrangement* ba)
    : ba_(ba)
{

}

FluidImplCuda::~FluidImplCuda()
{
}

bool staggered = true;

void FluidImplCuda::Advect(cudaArray* dest, cudaArray* velocity,
                           cudaArray* source, cudaArray* intermediate,
                           float time_step, float dissipation,
                           const glm::ivec3& volume_size,
                           AdvectionMethod method)
{
    if (staggered)
        LaunchAdvectScalarStaggered(dest, velocity, source, intermediate,
                                    time_step, dissipation, false,
                                    FromGlmVector(volume_size), method);
    else
        LaunchAdvectScalar(dest, velocity, source, intermediate, time_step,
                           dissipation, false, FromGlmVector(volume_size),
                           method);
}

void FluidImplCuda::AdvectDensity(cudaArray* dest, cudaArray* velocity,
                                  cudaArray* density, cudaArray* intermediate,
                                  float time_step, float dissipation,
                                  const glm::ivec3& volume_size,
                                  AdvectionMethod method)
{
    if (staggered)
        LaunchAdvectScalarStaggered(dest, velocity, density, intermediate,
                                    time_step, dissipation, true,
                                    FromGlmVector(volume_size), method);
    else
        LaunchAdvectScalar(dest, velocity, density, intermediate, time_step,
                           dissipation, true, FromGlmVector(volume_size),
                           method);
}

void FluidImplCuda::AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                                   cudaArray* velocity_prev, float time_step,
                                   float time_step_prev, float dissipation,
                                   const glm::ivec3& volume_size,
                                   AdvectionMethod method)
{
    if (staggered)
        LaunchAdvectVelocityStaggered(dest, velocity, velocity_prev, time_step,
                                      time_step_prev, dissipation,
                                      FromGlmVector(volume_size), method);
    else
        LaunchAdvectVelocity(dest, velocity, velocity_prev, time_step,
                             time_step_prev, dissipation,
                             FromGlmVector(volume_size), method);
}

void FluidImplCuda::ApplyBuoyancy(cudaArray* dest, cudaArray* velocity,
                                  cudaArray* temperature, float time_step,
                                  float ambient_temperature,
                                  float accel_factor, float gravity,
                                  const glm::ivec3& volume_size)
{
    if (staggered)
        LaunchApplyBuoyancyStaggered(dest, velocity, temperature, time_step,
                                     ambient_temperature, accel_factor, gravity,
                                     FromGlmVector(volume_size));
    else
        LaunchApplyBuoyancy(dest, velocity, temperature, time_step,
                            ambient_temperature, accel_factor, gravity,
                            FromGlmVector(volume_size));
}

void FluidImplCuda::ApplyImpulse(cudaArray* dest, cudaArray* source,
                                 const glm::vec3& center_point,
                                 const glm::vec3& hotspot, float radius,
                                 const glm::vec3& value, uint32_t mask,
                                 const glm::ivec3& volume_size)
{
    LaunchApplyImpulse(
        dest, source,
        make_float3(center_point.x, center_point.y, center_point.z),
        make_float3(hotspot.x, hotspot.y, hotspot.z),
        radius, make_float3(value.x, value.y, value.z), mask,
        FromGlmVector(volume_size));
}

void FluidImplCuda::ApplyImpulseDensity(cudaArray* density,
                                        const glm::vec3& center_point,
                                        const glm::vec3& hotspot,
                                        float radius, float value,
                                        const glm::ivec3& volume_size)
{
    LaunchApplyImpulse(
        density, density,
        make_float3(center_point.x, center_point.y, center_point.z),
        make_float3(hotspot.x, hotspot.y, hotspot.z),
        radius, make_float3(value, 0, 0), 1, FromGlmVector(volume_size));
}

void FluidImplCuda::ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                                      float half_inverse_cell_size,
                                      const glm::ivec3& volume_size)
{
    if (staggered)
        LaunchComputeDivergenceStaggered(dest, velocity,
                                         2.0f * half_inverse_cell_size,
                                         FromGlmVector(volume_size));
    else
        LaunchComputeDivergence(dest, velocity, half_inverse_cell_size,
                                FromGlmVector(volume_size));
}

void FluidImplCuda::ComputeResidualPackedDiagnosis(
    cudaArray* dest, cudaArray* source, float inverse_h_square,
    const glm::ivec3& volume_size)
{
    LaunchComputeResidualPackedDiagnosis(dest, source, inverse_h_square,
                                         FromGlmVector(volume_size));
}

void FluidImplCuda::DampedJacobi(cudaArray* dest, cudaArray* source,
                                 float minus_square_cell_size,
                                 float omega_over_beta, int num_of_iterations,
                                 const glm::ivec3& volume_size)
{
    LaunchDampedJacobi(dest, source, minus_square_cell_size, omega_over_beta,
                       num_of_iterations, FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ReviseDensity(cudaArray* density,
                                  const glm::vec3& center_point, float radius,
                                  float value, const glm::ivec3& volume_size)
{
    LaunchImpulseDensity(
        density, density,
        make_float3(center_point.x, center_point.y, center_point.z),
        radius, make_float3(value, 0, 0), FromGlmVector(volume_size));
}

void FluidImplCuda::SubtractGradient(cudaArray* dest, cudaArray* packed,
                                     float half_inverse_cell_size,
                                     const glm::ivec3& volume_size)
{
    if (staggered)
        LaunchSubtractGradientStaggered(dest, packed,
                                        2.0f * half_inverse_cell_size,
                                        FromGlmVector(volume_size), ba_);
    else
        LaunchSubtractGradient(dest, packed, half_inverse_cell_size,
                               FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::RoundPassed(int round)
{
    int* dest_array = nullptr;
    cudaError_t result = cudaMalloc(&dest_array, 4);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchRoundPassed(dest_array, round, 3);

    cudaFree(dest_array);
}
