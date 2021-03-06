//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include "fluid_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>
#include <driver_types.h>

#include "aux_buffer_manager.h"
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

FluidImplCuda::FluidImplCuda(BlockArrangement* ba)
    : ba_(ba)
    , cell_size_(0.15f)
    , staggered_(true)
    , mid_point_(false)
    , outflow_(false)
    , advect_method_(MACCORMACK_SEMI_LAGRANGIAN)
    , impulse_(IMPULSE_HOT_FLOOR)
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
        kern_launcher::AdvectScalarFieldStaggered(fnp1, fn, vel_x, vel_y, vel_z,
                                                  aux, cell_size_, time_step,
                                                  dissipation, advect_method_,
                                                  FromGlmVector(volume_size),
                                                  mid_point_, ba_);
    else
        kern_launcher::AdvectScalarField(fnp1, fn, vel_x, vel_y, vel_z, aux,
                                         cell_size_, time_step, dissipation,
                                         advect_method_,
                                         FromGlmVector(volume_size), mid_point_,
                                         ba_);
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
    kern_launcher::DecayVelocity(vel_x, vel_y, vel_z, time_step, dissipation,
                                 FromGlmVector(volume_size), ba_);
    if (staggered_) {
        if (field == VECTOR_FIELD_VELOCITY) {
            kern_launcher::AdvectVelocityStaggered(fnp1_x, fnp1_y, fnp1_z, fn_x,
                                                   fn_y, fn_z, vel_x, vel_y,
                                                   vel_z, aux, cell_size_,
                                                   time_step, 0.0f,
                                                   advect_method_,
                                                   FromGlmVector(volume_size),
                                                   mid_point_, ba_);
        } else if (field == VECTOR_FIELD_VORTICITY) {
            kern_launcher::AdvectVorticityStaggered(fnp1_x, fnp1_y, fnp1_z,
                                                    fn_x, fn_y, fn_z, vel_x,
                                                    vel_y, vel_z, aux,
                                                    cell_size_, time_step,
                                                    dissipation, advect_method_,
                                                    FromGlmVector(volume_size),
                                                    mid_point_, ba_);
        }
    } else {
        kern_launcher::AdvectVectorField(fnp1_x, fnp1_y, fnp1_z, fn_x, fn_y,
                                         fn_z, vel_x, vel_y, vel_z, aux,
                                         cell_size_, time_step, 0.0f,
                                         advect_method_,
                                         FromGlmVector(volume_size), mid_point_,
                                         ba_);
    }
}

void FluidImplCuda::ApplyBuoyancy(cudaArray* vnp1_x, cudaArray* vnp1_y,
                                  cudaArray* vnp1_z, cudaArray* vn_x,
                                  cudaArray* vn_y, cudaArray* vn_z,
                                  cudaArray* temperature, cudaArray* density,
                                  float time_step, float ambient_temperature,
                                  float accel_factor, float gravity,
                                  const glm::ivec3& volume_size)
{
    kern_launcher::ApplyBuoyancy(vnp1_x, vnp1_y, vnp1_z, vn_x, vn_y, vn_z,
                                 temperature, density, time_step,
                                 ambient_temperature, accel_factor, gravity,
                                 staggered_, FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ApplyImpulse(cudaArray* vnp1_x,cudaArray* vnp1_y,
                                 cudaArray* vnp1_z, cudaArray* d_np1,
                                 cudaArray* t_np1, cudaArray* vel_x,
                                 cudaArray* vel_y, cudaArray* vel_z,
                                 cudaArray* density, cudaArray* temperature,
                                 const glm::vec3& center_point,
                                 const glm::vec3& hotspot, float radius,
                                 const glm::vec3& vel_value, float d_value,
                                 float t_value, const glm::ivec3& volume_size)
{
    kern_launcher::ImpulseScalar(
        d_np1, density,
        make_float3(center_point.x, center_point.y, center_point.z),
        make_float3(hotspot.x, hotspot.y, hotspot.z),
        radius, d_value, impulse_, FromGlmVector(volume_size), ba_);

    if (std::abs(t_value) > 0.000001f)
        kern_launcher::ImpulseScalar(
            t_np1, temperature,
            make_float3(center_point.x, center_point.y, center_point.z),
            make_float3(hotspot.x, hotspot.y, hotspot.z), radius,
            t_value, impulse_, FromGlmVector(volume_size), ba_);

    kern_launcher::ImpulseVelocity(
        vnp1_x, vnp1_y, vnp1_z,
        make_float3(center_point.x, center_point.y, center_point.z),
        radius, make_float3(vel_value.x, vel_value.y, vel_value.z), impulse_,
        FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ApplyVorticityConfinement(cudaArray* vel_x,
                                              cudaArray* vel_y,
                                              cudaArray* vel_z,
                                              cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              const glm::ivec3& volume_size)
{
    kern_launcher::ApplyVorticityConfinementStaggered(
        vel_x, vel_y, vel_z, conf_x, conf_y, conf_z, FromGlmVector(volume_size),
        ba_);
}

void FluidImplCuda::BuildVorticityConfinement(cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              cudaArray* vort_x,
                                              cudaArray* vort_y,
                                              cudaArray* vort_z, float coeff,
                                              const glm::ivec3& volume_size)
{
    kern_launcher::BuildVorticityConfinementStaggered(
        conf_x, conf_y, conf_z, vort_x, vort_y, vort_z, coeff, cell_size_,
        FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeCurl(cudaArray* vort_x, cudaArray* vort_y,
                                cudaArray* vort_z, cudaArray* vel_x,
                                cudaArray* vel_y, cudaArray* vel_z,
                                const glm::ivec3& volume_size)
{
    kern_launcher::ComputeCurlStaggered(vort_x, vort_y, vort_z, vel_x, vel_y,
                                        vel_z, cell_size_,
                                        FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeDivergence(cudaArray* div, cudaArray* vel_x,
                                      cudaArray* vel_y, cudaArray* vel_z,
                                      const glm::ivec3& volume_size)
{
    kern_launcher::ComputeDivergence(div, vel_x, vel_y, vel_z, cell_size_,
                                     outflow_, staggered_,
                                     FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeResidualDiagnosis(cudaArray* residual, cudaArray* u,
                                             cudaArray* b,
                                             const glm::ivec3& volume_size)
{
    kern_launcher::ComputeResidualDiagnosis(residual, u, b, cell_size_,
                                            FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::Relax(cudaArray* unp1, cudaArray* un, cudaArray* b,
                          int num_of_iterations, const glm::ivec3& volume_size)
{
    kern_launcher::Relax(unp1, un, b, outflow_, num_of_iterations,
                         FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ReviseDensity(cudaArray* density,
                                  const glm::vec3& center_point, float radius,
                                  float value, const glm::ivec3& volume_size)
{
    kern_launcher::ImpulseDensity(
        density, density,
        make_float3(center_point.x, center_point.y, center_point.z), radius,
        value, impulse_, FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::SubtractGradient(cudaArray* vel_x, cudaArray* vel_y,
                                     cudaArray* vel_z, cudaArray* pressure,
                                     const glm::ivec3& volume_size)
{
    kern_launcher::SubtractGradient(vel_x, vel_y, vel_z, pressure, cell_size_,
                                    staggered_, FromGlmVector(volume_size),
                                    ba_);
}

void FluidImplCuda::AddCurlPsi(cudaArray* vel_x, cudaArray* vel_y,
                               cudaArray* vel_z, cudaArray* psi_x,
                               cudaArray* psi_y, cudaArray* psi_z,
                               const glm::ivec3& volume_size)
{
    kern_launcher::AddCurlPsi(vel_x, vel_y, vel_z, psi_x, psi_y, psi_z,
                              cell_size_, FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::ComputeDeltaVorticity(cudaArray* delta_x,
                                          cudaArray* delta_y,
                                          cudaArray* delta_z,
                                          cudaArray* vort_x,
                                          cudaArray* vort_y, cudaArray* vort_z,
                                          const glm::ivec3& volume_size)
{
    kern_launcher::ComputeDeltaVorticity(delta_x, delta_y, delta_z, vort_x,
                                         vort_y, vort_z,
                                         FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::DecayVortices(cudaArray* vort_x, cudaArray* vort_y,
                                  cudaArray* vort_z, cudaArray* div,
                                  float time_step,
                                  const glm::ivec3& volume_size)
{
    kern_launcher::DecayVorticesStaggered(vort_x, vort_y, vort_z, div,
                                          time_step, FromGlmVector(volume_size),
                                          ba_);
}

void FluidImplCuda::StretchVortices(cudaArray* vnp1_x, cudaArray* vnp1_y,
                                    cudaArray* vnp1_z, cudaArray* vel_x,
                                    cudaArray* vel_y, cudaArray* vel_z,
                                    cudaArray* vort_x, cudaArray* vort_y,
                                    cudaArray* vort_z, float time_step,
                                    const glm::ivec3& volume_size)
{
    kern_launcher::StretchVorticesStaggered(vnp1_x, vnp1_y, vnp1_z, vel_x,
                                            vel_y, vel_z, vort_x, vort_y,
                                            vort_z, cell_size_, time_step,
                                            FromGlmVector(volume_size), ba_);
}

void FluidImplCuda::RoundPassed(int round)
{
    int* dest_array = nullptr;
    cudaError_t result = cudaMalloc(&dest_array, 4);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    kern_launcher::RoundPassed(dest_array, round, 3);

    cudaFree(dest_array);
}
