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

#ifndef _FLUID_IMPL_CUDA_H_
#define _FLUID_IMPL_CUDA_H_

#include <memory>

#include "advection_method.h"
#include "fluid_impulse.h"
#include "third_party/glm/fwd.hpp"

struct cudaArray;
class BlockArrangement;
class CudaVolume;
class GraphicsResource;
class FluidImplCuda
{
public:
    enum VectorField
    {
        VECTOR_FIELD_VELOCITY,
        VECTOR_FIELD_VORTICITY,
    };

    explicit FluidImplCuda(BlockArrangement* ba);
    ~FluidImplCuda();

    void AdvectScalarField(cudaArray* fnp1, cudaArray* fn, cudaArray* vel_x,
                           cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux,
                           float time_step, float dissipation,
                           const glm::ivec3& volume_size);
    void AdvectVectorFields(cudaArray* fnp1_x, cudaArray* fnp1_y,
                            cudaArray* fnp1_z, cudaArray* fn_x, cudaArray* fn_y,
                            cudaArray* fn_z, cudaArray* vel_x, cudaArray* vel_y,
                            cudaArray* vel_z, cudaArray* aux, float time_step,
                            float dissipation, const glm::ivec3& volume_size,
                            VectorField field);
    void ApplyBuoyancy(cudaArray* vnp1_x, cudaArray* vnp1_y, cudaArray* vnp1_z,
                       cudaArray* vn_x, cudaArray* vn_y, cudaArray* vn_z,
                       cudaArray* temperature, cudaArray* density,
                       float time_step, float ambient_temperature,
                       float accel_factor, float gravity,
                       const glm::ivec3& volume_size);
    void ApplyImpulse(cudaArray* vnp1_x, cudaArray* vnp1_y, cudaArray* vnp1_z,
                      cudaArray* d_np1, cudaArray* t_np1, cudaArray* vel_x,
                      cudaArray* vel_y, cudaArray* vel_z, cudaArray* density,
                      cudaArray* temperature, const glm::vec3& center_point,
                      const glm::vec3& hotspot, float radius,
                      const glm::vec3& vel_value, float d_value, float t_value,
                      const glm::ivec3& volume_size);
    void ApplyVorticityConfinement(cudaArray* vel_x, cudaArray* vel_y,
                                   cudaArray* vel_z, cudaArray* conf_x,
                                   cudaArray* conf_y, cudaArray* conf_z,
                                   const glm::ivec3& volume_size);
    void BuildVorticityConfinement(cudaArray* conf_x, cudaArray* conf_y,
                                   cudaArray* conf_z, cudaArray* vort_x,
                                   cudaArray* vort_y, cudaArray* vort_z,
                                   float coeff, const glm::ivec3& volume_size);
    void ComputeCurl(cudaArray* vort_x, cudaArray* vort_y, cudaArray* vort_z,
                     cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                     const glm::ivec3& volume_size);
    void ComputeDivergence(cudaArray* div, cudaArray* vel_x,
                           cudaArray* vel_y, cudaArray* vel_z,
                           const glm::ivec3& volume_size);
    void ComputeResidualDiagnosis(cudaArray* residual, cudaArray* u,
                                  cudaArray* b, const glm::ivec3& volume_size);
    void Relax(cudaArray* unp1, cudaArray* un, cudaArray* b,
               int num_of_iterations, const glm::ivec3& volume_size);
    void ReviseDensity(cudaArray* density, const glm::vec3& center_point,
                       float radius, float value,
                       const glm::ivec3& volume_size);
    void SubtractGradient(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                          cudaArray* pressure, const glm::ivec3& volume_size);

    // Vorticity.
    void AddCurlPsi(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                    cudaArray* psi_x, cudaArray* psi_y, cudaArray* psi_z,
                    const glm::ivec3& volume_size);
    void ComputeDeltaVorticity(cudaArray* delta_x, cudaArray* delta_y,
                               cudaArray* delta_z, cudaArray* vort_x,
                               cudaArray* vort_y, cudaArray* vort_z,
                               const glm::ivec3& volume_size);
    void DecayVortices(cudaArray* vort_x, cudaArray* vort_y, cudaArray* vort_z,
                       cudaArray* div, float time_step,
                       const glm::ivec3& volume_size);
    void StretchVortices(cudaArray* vnp1_x, cudaArray* vnp1_y,
                         cudaArray* vnp1_z, cudaArray* vel_x, cudaArray* vel_y,
                         cudaArray* vel_z, cudaArray* vort_x, cudaArray* vort_y,
                         cudaArray* vort_z, float time_step,
                         const glm::ivec3& volume_size);

    // For debugging.
    void RoundPassed(int round);

    void set_cell_size(float cell_size) { cell_size_ = cell_size; }
    void set_advect_method(AdvectionMethod m) { advect_method_ = m; }
    void set_fluid_impulse(FluidImpulse i) { impulse_ = i; }
    void set_mid_point(bool mid_point) { mid_point_ = mid_point; }
    void set_outflow(bool outflow) { outflow_ = outflow; }
    void set_staggered(bool staggered) { staggered_ = staggered; }

private:
    BlockArrangement* ba_;
    float cell_size_;
    bool staggered_;
    bool mid_point_;
    bool outflow_;
    AdvectionMethod advect_method_;
    FluidImpulse impulse_;
};

#endif // _FLUID_IMPL_CUDA_H_