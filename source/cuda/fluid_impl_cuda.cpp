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
#include "kernel_launcher.h"
#include "third_party/glm/vec3.hpp"

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
    , mid_point_(false)
    , advect_method_(MACCORMACK_SEMI_LAGRANGIAN)
{

}

FluidImplCuda::~FluidImplCuda()
{
}

void FluidImplCuda::AdvectScalarField(cudaArray* fnp1, cudaArray* fn,
                                      cudaArray* vel_x, cudaArray* vel_y,
                                      cudaArray* vel_z, cudaArray* aux,
                                      float time_step, float dissipation,
                                      const glm::ivec3& volume_size)
{
    if (staggered_)
        LaunchAdvectScalarFieldStaggered(fnp1, fn, vel_x, vel_y, vel_z, aux,
                                         time_step, dissipation,
                                         advect_method_,
                                         FromGlmVector(volume_size), mid_point_,
                                         ba_);
    else
        LaunchAdvectScalarField(fnp1, fn, vel_x, vel_y, vel_z, aux, time_step,
                                dissipation, advect_method_,
                                FromGlmVector(volume_size), mid_point_, ba_);
}

void FluidImplCuda::AdvectVectorFields(cudaArray* fnp1_x, cudaArray* fnp1_y,
                                       cudaArray* fnp1_z, cudaArray* fn_x,
                                       cudaArray* fn_y, cudaArray* fn_z,
                                       cudaArray* vel_x, cudaArray* vel_y,
                                       cudaArray* vel_z, cudaArray* aux,
                                       float time_step, float dissipation,
                                       const glm::ivec3& volume_size,
                                       VectorField field)
{
    if (staggered_) {
        if (field == VECTOR_FIELD_VELOCITY) {
            LaunchAdvectVelocityStaggered(fnp1_x, fnp1_y, fnp1_z, fn_x, fn_y,
                                          fn_z, vel_x, vel_y, vel_z, aux,
                                          time_step, dissipation,
                                          advect_method_,
                                          FromGlmVector(volume_size),
                                          mid_point_, ba_);
        } else if (field == VECTOR_FIELD_VORTICITY) {
            LaunchAdvectVorticityStaggered(fnp1_x, fnp1_y, fnp1_z, fn_x, fn_y,
                                           fn_z, vel_x, vel_y, vel_z, aux,
                                           time_step, dissipation,
                                           advect_method_,
                                           FromGlmVector(volume_size),
                                           mid_point_, ba_);
        }
    } else {
        LaunchAdvectVectorField(fnp1_x, fnp1_y, fnp1_z, fn_x, fn_y, fn_z, vel_x,
                                vel_y, vel_z, aux, time_step, dissipation,
                                advect_method_, FromGlmVector(volume_size),
                                mid_point_, ba_);
    }
}

void FluidImplCuda::ApplyBuoyancy(cudaArray* vel_x, cudaArray* vel_y,
                                  cudaArray* vel_z, cudaArray* temperature,
                                  cudaArray* density, float time_step,
                                  float ambient_temperature, float accel_factor,
                                  float gravity, const glm::ivec3& volume_size)
{
    if (staggered_)
        LaunchApplyBuoyancyStaggered(vel_x, vel_y, vel_z, temperature, density,
                                     time_step, ambient_temperature,
                                     accel_factor, gravity,
                                     FromGlmVector(volume_size), ba_);
    else
        LaunchApplyBuoyancy(vel_x, vel_y, vel_z, temperature, density,
                            time_step, ambient_temperature, accel_factor,
                            gravity, FromGlmVector(volume_size), ba_);
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
            FromGlmVector(volume_size), ba_);
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
            radius, make_float3(value, 0, 0), 1, FromGlmVector(volume_size),
            ba_);
}

void FluidImplCuda::ApplyVorticityConfinement(cudaArray* vel_x,
                                              cudaArray* vel_y,
                                              cudaArray* vel_z,
                                              cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              const glm::ivec3& volume_size)
{
    LaunchApplyVorticityConfinementStaggered(vel_x, vel_y, vel_z, conf_x,
                                             conf_y, conf_z,
                                             FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::BuildVorticityConfinement(cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              cudaArray* vort_x,
                                              cudaArray* vort_y,
                                              cudaArray* vort_z,
                                              float coeff, float cell_size,
                                              const glm::ivec3& volume_size)
{
    LaunchBuildVorticityConfinementStaggered(conf_x, conf_y, conf_z, vort_x,
                                             vort_y, vort_z, coeff, cell_size,
                                             FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeCurl(cudaArray* vort_x, cudaArray* vort_y,
                                cudaArray* vort_z, cudaArray* vel_x,
                                cudaArray* vel_y, cudaArray* vel_z,
                                float cell_size, const glm::ivec3& volume_size)
{
    LaunchComputeCurlStaggered(vort_x, vort_y, vort_z, vel_x, vel_y,
                               vel_z, cell_size, FromGlmVector(volume_size),
                               ba_);
}

void FluidImplCuda::ComputeDivergence(cudaArray* div, cudaArray* vel_x,
                                      cudaArray* vel_y, cudaArray* vel_z,
                                      float cell_size,
                                      const glm::ivec3& volume_size)
{
    if (staggered_)
        LaunchComputeDivergenceStaggered(div, vel_x, vel_y, vel_z,
                                         cell_size,
                                         FromGlmVector(volume_size), ba_);
    else
        LaunchComputeDivergence(div, vel_x, vel_y, vel_z, cell_size,
                                FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeResidualDiagnosis(cudaArray* residual, cudaArray* u,
                                             cudaArray* b,
                                             float cell_size,
                                             const glm::ivec3& volume_size)
{
    LaunchComputeResidualDiagnosis(residual, u, b, cell_size,
                                   FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::Relax(cudaArray* unp1, cudaArray* un, cudaArray* b,
                          float cell_size, int num_of_iterations,
                          const glm::ivec3& volume_size)
{
    LaunchRelax(unp1, un, b, cell_size, num_of_iterations,
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
            radius, make_float3(value, 0, 0), FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::SubtractGradient(cudaArray* vel_x, cudaArray* vel_y,
                                     cudaArray* vel_z, cudaArray* pressure,
                                     float cell_size,
                                     const glm::ivec3& volume_size)
{
    if (staggered_)
        LaunchSubtractGradientStaggered(vel_x, vel_y, vel_z, pressure,
                                        cell_size, FromGlmVector(volume_size),
                                        ba_);
    else
        LaunchSubtractGradient(vel_x, vel_y, vel_z, pressure,
                               cell_size, FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::AddCurlPsi(cudaArray* vel_x, cudaArray* vel_y,
                               cudaArray* vel_z, cudaArray* psi_x,
                               cudaArray* psi_y, cudaArray* psi_z,
                               float cell_size, const glm::ivec3& volume_size)
{
    LaunchAddCurlPsi(vel_x, vel_y, vel_z, psi_x, psi_y, psi_z, cell_size,
                     FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeDeltaVorticity(cudaArray* delta_x,
                                          cudaArray* delta_y,
                                          cudaArray* delta_z,
                                          cudaArray* vort_x,
                                          cudaArray* vort_y, cudaArray* vort_z,
                                          const glm::ivec3& volume_size)
{
    LaunchComputeDeltaVorticity(delta_x, delta_y, delta_z, vort_x,
                                vort_y, vort_z, FromGlmVector(volume_size),
                                ba_);
}

void FluidImplCuda::DecayVortices(cudaArray* vort_x, cudaArray* vort_y,
                                  cudaArray* vort_z, cudaArray* div,
                                  float time_step,
                                  const glm::ivec3& volume_size)
{
    LaunchDecayVorticesStaggered(vort_x, vort_y, vort_z, div, time_step,
                                 FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::StretchVortices(cudaArray* vnp1_x, cudaArray* vnp1_y,
                                    cudaArray* vnp1_z, cudaArray* vel_x,
                                    cudaArray* vel_y, cudaArray* vel_z,
                                    cudaArray* vort_x, cudaArray* vort_y,
                                    cudaArray* vort_z, float cell_size,
                                    float time_step,
                                    const glm::ivec3& volume_size)
{
    LaunchStretchVorticesStaggered(vnp1_x, vnp1_y, vnp1_z, vel_x, vel_y, vel_z,
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
