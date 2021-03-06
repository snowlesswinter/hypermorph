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

#ifndef _GRID_FLUID_SOLVER_H_
#define _GRID_FLUID_SOLVER_H_

#include <memory>

#include "fluid_field_owner.h"
#include "fluid_solver.h"
#include "poisson_solver/poisson_solver_enum.h"
#include "third_party/glm/vec3.hpp"

class GraphicsVolume;
class GraphicsVolume3;
class PoissonCore;
class PoissonSolver;
class GridFluidSolver : public FluidSolver, public FluidFieldOwner
{
public:
    GridFluidSolver();
    virtual ~GridFluidSolver();

    // Overridden from FluidSolver:
    virtual void Impulse(float splat_radius, const glm::vec3& impulse_position,
                         const glm::vec3& hotspot, float impulse_density,
                         float impulse_temperature,
                         const glm::vec3& impulse_velocity) override;
    virtual bool Initialize(GraphicsLib graphics_lib, int width, int height,
                            int depth, int poisson_byte_width) override;
    virtual void Reset() override;
    virtual void SetDiagnosis(int diagnosis) override;
    virtual void SetPressureSolver(PoissonSolver* solver) override;
    virtual void Solve(float delta_time) override;

    // Overridden from FluidFieldOwner:
    virtual GraphicsVolume* GetDensityField() override;
    virtual GraphicsVolume3* GetVelocityField() override;
    virtual GraphicsVolume* GetTemperatureField() override;

private:
    friend class FluidUnittest;

    void AdvectDensity(float delta_time);
    void AdvectImpl(const GraphicsVolume& source, float delta_time,
                    float dissipation);
    void AdvectTemperature(float delta_time);
    void AdvectVelocity(float delta_time);
    void ApplyBuoyancy(float delta_time);
    void ComputeDivergence(std::shared_ptr<GraphicsVolume> divergence);
    void ComputeResidualDiagnosis(std::shared_ptr<GraphicsVolume> pressure,
                                  std::shared_ptr<GraphicsVolume> divergence);
    void DampedJacobi(std::shared_ptr<GraphicsVolume> pressure,
                      std::shared_ptr<GraphicsVolume> divergence,
                      float cell_size, int num_of_iterations);
    void ImpulseField(std::shared_ptr<GraphicsVolume> dest,
                      const glm::vec3& position, const glm::vec3& hotspot,
                      float splat_radius, float value);
    void ImpulseDensity(const glm::vec3& position, const glm::vec3& hotspot,
                        float splat_radius,
                        float value);
    void ReviseDensity();
    void SolvePressure(std::shared_ptr<GraphicsVolume> pressure,
                       std::shared_ptr<GraphicsVolume> divergence);
    void SubtractGradient(std::shared_ptr<GraphicsVolume> pressure);

    // Vorticity.
    void AddCurlPsi(const GraphicsVolume3& psi);
    void AdvectVortices(const GraphicsVolume3& vorticity,
                        const GraphicsVolume3& temp,
                        std::shared_ptr<GraphicsVolume> aux, float delta_time);
    void ApplyVorticityConfinemnet();
    void BuildVorticityConfinemnet(float delta_time);
    void ComputeCurl(const GraphicsVolume3& vorticity,
                     const GraphicsVolume3& velocity);
    void ComputeDeltaVorticity(const GraphicsVolume3& aux,
                               const GraphicsVolume3& vorticity);
    void DecayVortices(const GraphicsVolume3& vorticity,
                       std::shared_ptr<GraphicsVolume> aux, float delta_time);
    void RestoreVorticity(float delta_time);
    void SolvePsi(const GraphicsVolume3& psi,
                  const GraphicsVolume3& delta_vort, int num_iterations);
    void StretchVortices(const GraphicsVolume3& vort_np1,
                         const GraphicsVolume3& vorticity, float delta_time);

    const GraphicsVolume3& GetVorticityField();
    const GraphicsVolume3& GetAuxField();
    const GraphicsVolume3& GetVorticityConfinementField();

    GraphicsLib graphics_lib_;
    glm::ivec3 grid_size_;
    PoissonSolver* pressure_solver_;
    int diagnosis_;

    std::shared_ptr<GraphicsVolume3> velocity_;
    std::shared_ptr<GraphicsVolume3> velocity_prime_;
    std::shared_ptr<GraphicsVolume3> vorticity_;
    std::shared_ptr<GraphicsVolume3> aux_;
    std::shared_ptr<GraphicsVolume3> vort_conf_;
    std::shared_ptr<GraphicsVolume> density_;
    std::shared_ptr<GraphicsVolume> temperature_;
    std::shared_ptr<GraphicsVolume> general1a_;
    std::shared_ptr<GraphicsVolume> general1b_;
    std::shared_ptr<GraphicsVolume> general1c_;
    std::shared_ptr<GraphicsVolume> general1d_;
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;

    bool need_buoyancy_;
    int frame_;
};

#endif // _GRID_FLUID_SOLVER_H_