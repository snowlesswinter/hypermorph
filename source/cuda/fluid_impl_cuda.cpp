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

extern void LaunchAdvectFieldsStaggered(cudaArray* fnp1_x, cudaArray* fnp1_y,
                                        cudaArray* fnp1_z, cudaArray* fn_x,
                                        cudaArray* fn_y, cudaArray* fn_z,
                                        cudaArray* aux, cudaArray* velocity,
                                        float time_step, float dissipation,
                                        uint3 volume_size, BlockArrangement* ba,
                                        AdvectionMethod method);
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
                                cudaArray* density_array, float time_step,
                                float ambient_temperature, float accel_factor,
                                float gravity, uint3 volume_size);
extern void LaunchApplyBuoyancyStaggered(cudaArray* dest_array,
                                         cudaArray* velocity_array,
                                         cudaArray* temperature_array,
                                         cudaArray* density_array,
                                         float time_step,
                                         float ambient_temperature,
                                         float accel_factor, float gravity,
                                         uint3 volume_size);
extern void LaunchApplyImpulse(cudaArray* dest_array, cudaArray* original_array,
                               float3 center_point, float3 hotspot,
                               float radius, float3 value, uint32_t mask,
                               uint3 volume_size);
extern void LaunchApplyVorticityConfinementStaggered(cudaArray* dest,
                                                     cudaArray* velocity,
                                                     cudaArray* conf_x,
                                                     cudaArray* conf_y,
                                                     cudaArray* conf_z,
                                                     uint3 volume_size,
                                                     BlockArrangement* ba);
extern void LaunchBuildVorticityConfinementStaggered(cudaArray* dest_x,
                                                     cudaArray* dest_y,
                                                     cudaArray* dest_z,
                                                     cudaArray* curl_x,
                                                     cudaArray* curl_y,
                                                     cudaArray* curl_z,
                                                     float coeff,
                                                     float cell_size,
                                                     uint3 volume_size,
                                                     BlockArrangement* ba);
extern void LaunchComputeCurlStaggered(cudaArray* dest_x, cudaArray* dest_y,
                                       cudaArray* dest_z, cudaArray* velocity,
                                       cudaArray* curl_x, cudaArray* curl_y,
                                       cudaArray* curl_z,
                                       float inverse_cell_size,
                                       uint3 volume_size, BlockArrangement* ba);
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
extern void LaunchGenerateHeatSphere(cudaArray* dest, cudaArray* original,
                                     float3 center_point, float radius,
                                     float3 value, uint3 volume_size,
                                     BlockArrangement* ba);
extern void LaunchImpulseDensity(cudaArray* dest_array,
                                 cudaArray* original_array, float3 center_point,
                                 float radius, float3 value, uint3 volume_size);
extern void LaunchImpulseDensitySphere(cudaArray* dest, cudaArray* original,
                                       float3 center_point, float radius,
                                       float3 value, uint3 volume_size,
                                       BlockArrangement* ba);
extern void LaunchRelax(cudaArray* dest, cudaArray* source, float cell_size,
                        int num_of_iterations, uint3 volume_size,
                        BlockArrangement* ba);
extern void LaunchRelax2(cudaArray* unp1, cudaArray* un, cudaArray* b,
                         float cell_size, int num_of_iterations,
                         uint3 volume_size, BlockArrangement* ba);
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

// Vorticity.
extern void LaunchAddCurlPsi(cudaArray* velocity, cudaArray* psi_x,
                             cudaArray* psi_y, cudaArray* psi_z,
                             float cell_size, uint3 volume_size,
                             BlockArrangement* ba);
extern void LaunchAdvectVorticityStaggered(cudaArray* fnp1_x, cudaArray* fnp1_y,
                                           cudaArray* fnp1_z, cudaArray* fn_x,
                                           cudaArray* fn_y, cudaArray* fn_z,
                                           cudaArray* aux, cudaArray* velocity,
                                           float time_step, float dissipation,
                                           uint3 volume_size,
                                           BlockArrangement* ba,
                                           AdvectionMethod method);
extern void LaunchComputeDivergenceStaggeredForVort(cudaArray* div,
                                                    cudaArray* velocity,
                                                    float cell_size,
                                                    uint3 volume_size);
extern void LaunchComputeDeltaVorticity(cudaArray* vnp1_x, cudaArray* vnp1_y,
                                        cudaArray* vnp1_z, cudaArray* vn_x,
                                        cudaArray* vn_y, cudaArray* vn_z,
                                        uint3 volume_size,
                                        BlockArrangement* ba);
extern void LaunchDecayVorticesStaggered(cudaArray* vort_x, cudaArray* vort_y,
                                         cudaArray* vort_z, cudaArray* div,
                                         float time_step, uint3 volume_size,
                                         BlockArrangement* ba);
extern void LaunchStretchVorticesStaggered(cudaArray* vort_np1_x,
                                           cudaArray* vort_np1_y,
                                           cudaArray* vort_np1_z,
                                           cudaArray* velocity,
                                           cudaArray* vort_x, cudaArray* vort_y,
                                           cudaArray* vort_z, float cell_size,
                                           float time_step, uint3 volume_size,
                                           BlockArrangement* ba);

namespace
{
uint3 FromGlmVector(const glm::ivec3& v)
{
    return make_uint3(static_cast<uint>(v.x), static_cast<uint>(v.y),
                      static_cast<uint>(v.z));
}
} // Anonymous namespace.

extern int sphere;

FluidImplCuda::FluidImplCuda(BlockArrangement* ba)
    : ba_(ba)
    , staggered_(true)
{

}

FluidImplCuda::~FluidImplCuda()
{
}

void FluidImplCuda::Advect(cudaArray* dest, cudaArray* velocity,
                           cudaArray* source, cudaArray* intermediate,
                           float time_step, float dissipation,
                           const glm::ivec3& volume_size,
                           AdvectionMethod method)
{
    if (staggered_)
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
    if (staggered_)
        LaunchAdvectScalarStaggered(dest, velocity, density, intermediate,
                                    time_step, dissipation, true,
                                    FromGlmVector(volume_size), method);
    else
        LaunchAdvectScalar(dest, velocity, density, intermediate, time_step,
                           dissipation, true, FromGlmVector(volume_size),
                           method);
}

void FluidImplCuda::AdvectFields(cudaArray* fnp1_x, cudaArray* fnp1_y,
                                 cudaArray* fnp1_z, cudaArray* fn_x,
                                 cudaArray* fn_y, cudaArray* fn_z,
                                 cudaArray* aux, cudaArray* velocity,
                                 float time_step, float dissipation,
                                 const glm::ivec3& volume_size)
{
    LaunchAdvectFieldsStaggered(fnp1_x,  fnp1_y, fnp1_z,  fn_x, fn_y,  fn_z,
                                aux, velocity, time_step, dissipation,
                                FromGlmVector(volume_size), ba_,
                                MACCORMACK_SEMI_LAGRANGIAN);
}

void FluidImplCuda::AdvectVorticityFields(cudaArray* fnp1_x, cudaArray* fnp1_y,
                                          cudaArray* fnp1_z, cudaArray* fn_x,
                                          cudaArray* fn_y, cudaArray* fn_z,
                                          cudaArray* aux, cudaArray* velocity,
                                          float time_step, float dissipation,
                                          const glm::ivec3& volume_size)
{
    LaunchAdvectVorticityStaggered(fnp1_x, fnp1_y, fnp1_z, fn_x, fn_y, fn_z,
                                   aux, velocity, time_step, dissipation,
                                   FromGlmVector(volume_size), ba_,
                                   MACCORMACK_SEMI_LAGRANGIAN);
}

void FluidImplCuda::AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                                   cudaArray* velocity_prev, float time_step,
                                   float time_step_prev, float dissipation,
                                   const glm::ivec3& volume_size,
                                   AdvectionMethod method)
{
    if (staggered_)
        LaunchAdvectVelocityStaggered(dest, velocity, velocity_prev, time_step,
                                      time_step_prev, dissipation,
                                      FromGlmVector(volume_size), method);
    else
        LaunchAdvectVelocity(dest, velocity, velocity_prev, time_step,
                             time_step_prev, dissipation,
                             FromGlmVector(volume_size), method);
}

void FluidImplCuda::ApplyBuoyancy(cudaArray* dest, cudaArray* velocity,
                                  cudaArray* temperature, cudaArray* density,
                                  float time_step, float ambient_temperature,
                                  float accel_factor, float gravity,
                                  const glm::ivec3& volume_size)
{
    if (staggered_)
        LaunchApplyBuoyancyStaggered(dest, velocity, temperature, density,
                                     time_step, ambient_temperature,
                                     accel_factor, gravity,
                                     FromGlmVector(volume_size));
    else
        LaunchApplyBuoyancy(dest, velocity, temperature, density, time_step,
                            ambient_temperature, accel_factor, gravity,
                            FromGlmVector(volume_size));
}

void FluidImplCuda::ApplyImpulse(cudaArray* dest, cudaArray* source,
                                 const glm::vec3& center_point,
                                 const glm::vec3& hotspot, float radius,
                                 const glm::vec3& value, uint32_t mask,
                                 const glm::ivec3& volume_size)
{
    if (sphere)
        LaunchGenerateHeatSphere(
            dest, source,
            make_float3(center_point.x, center_point.y, center_point.z),
            radius, make_float3(value.x, value.y, value.z),
            FromGlmVector(volume_size), ba_);
    else
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
    if (sphere)
        LaunchImpulseDensitySphere(
            density, density,
            make_float3(center_point.x, center_point.y, center_point.z),
            radius, make_float3(value, value, value),
            FromGlmVector(volume_size), ba_);
    else
        LaunchApplyImpulse(
            density, density,
            make_float3(center_point.x, center_point.y, center_point.z),
            make_float3(hotspot.x, hotspot.y, hotspot.z),
            radius, make_float3(value, 0, 0), 1, FromGlmVector(volume_size));
}

void FluidImplCuda::ApplyVorticityConfinement(cudaArray* dest,
                                              cudaArray* velocity,
                                              cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              const glm::ivec3& volume_size)
{
    LaunchApplyVorticityConfinementStaggered(dest, velocity, conf_x, conf_y,
                                             conf_z, FromGlmVector(volume_size),
                                             ba_);
}

void FluidImplCuda::BuildVorticityConfinement(cudaArray* dest_x,
                                              cudaArray* dest_y,
                                              cudaArray* dest_z,
                                              cudaArray* curl_x,
                                              cudaArray* curl_y,
                                              cudaArray* curl_z,
                                              float coeff, float cell_size,
                                              const glm::ivec3& volume_size)
{
    LaunchBuildVorticityConfinementStaggered(dest_x, dest_y, dest_z, curl_x,
                                             curl_y, curl_z, coeff, cell_size,
                                             FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeCurl(cudaArray* dest_x, cudaArray* dest_y,
                                cudaArray* dest_z, cudaArray* velocity,
                                cudaArray* curl_x, cudaArray* curl_y,
                                cudaArray* curl_z, float inverse_cell_size,
                                const glm::ivec3& volume_size)
{
    LaunchComputeCurlStaggered(dest_x, dest_y, dest_z, velocity, curl_x, curl_y,
                               curl_z, inverse_cell_size,
                               FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                                      float half_inverse_cell_size,
                                      const glm::ivec3& volume_size)
{
    if (staggered_)
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

void FluidImplCuda::Relax(cudaArray* dest, cudaArray* source, float cell_size,
           int num_of_iterations, const glm::ivec3& volume_size)
{
    LaunchRelax(dest, source, cell_size, num_of_iterations,
                FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::Relax(cudaArray* unp1, cudaArray* un, cudaArray* b,
                          float cell_size, int num_of_iterations,
                          const glm::ivec3& volume_size)
{
    LaunchRelax2(unp1, un, b, cell_size, num_of_iterations,
                FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ReviseDensity(cudaArray* density,
                                  const glm::vec3& center_point, float radius,
                                  float value, const glm::ivec3& volume_size)
{
    if (sphere)
        LaunchImpulseDensitySphere(
            density, density,
            make_float3(center_point.x, center_point.y, center_point.z),
            radius, make_float3(value, 0, 0), FromGlmVector(volume_size), ba_);
    else
        LaunchImpulseDensity(
            density, density,
            make_float3(center_point.x, center_point.y, center_point.z),
            radius, make_float3(value, 0, 0), FromGlmVector(volume_size));
}

void FluidImplCuda::SubtractGradient(cudaArray* dest, cudaArray* packed,
                                     float half_inverse_cell_size,
                                     const glm::ivec3& volume_size)
{
    if (staggered_)
        LaunchSubtractGradientStaggered(dest, packed,
                                        2.0f * half_inverse_cell_size,
                                        FromGlmVector(volume_size), ba_);
    else
        LaunchSubtractGradient(dest, packed, half_inverse_cell_size,
                               FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::AddCurlPsi(cudaArray* velocity, cudaArray* psi_x,
                               cudaArray* psi_y, cudaArray* psi_z,
                               float cell_size, const glm::ivec3& volume_size)
{
    LaunchAddCurlPsi(velocity, psi_x, psi_y, psi_z, cell_size,
                     FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeDeltaVorticity(cudaArray* vort_np1_x,
                                          cudaArray* vort_np1_y,
                                          cudaArray* vort_np1_z,
                                          cudaArray* vort_x,
                                          cudaArray* vort_y, cudaArray* vort_z,
                                          const glm::ivec3& volume_size)
{
    LaunchComputeDeltaVorticity(vort_np1_x, vort_np1_y, vort_np1_z, vort_x,
                                vort_y, vort_z, FromGlmVector(volume_size),
                                ba_);
}

void FluidImplCuda::ComputeDivergenceForVort(cudaArray* div,
                                             cudaArray* velocity,
                                             float cell_size,
                                             const glm::ivec3& volume_size)
{
    LaunchComputeDivergenceStaggeredForVort(div, velocity, cell_size,
                                            FromGlmVector(volume_size));
}

void FluidImplCuda::DecayVortices(cudaArray* vort_x, cudaArray* vort_y,
                                  cudaArray* vort_z, cudaArray* div,
                                  float time_step,
                                  const glm::ivec3& volume_size)
{
    LaunchDecayVorticesStaggered(vort_x, vort_y, vort_z, div, time_step,
                                 FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::StretchVortices(cudaArray* vort_np1_x,
                                    cudaArray* vort_np1_y,
                                    cudaArray* vort_np1_z, cudaArray* velocity,
                                    cudaArray* vort_x, cudaArray* vort_y,
                                    cudaArray* vort_z, float cell_size,
                                    float time_step,
                                    const glm::ivec3& volume_size)
{
    LaunchStretchVorticesStaggered(vort_np1_x, vort_np1_y, vort_np1_z, velocity,
                                   vort_x, vort_y, vort_z, cell_size, time_step,
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
