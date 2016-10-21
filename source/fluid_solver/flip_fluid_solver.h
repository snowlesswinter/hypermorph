//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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

#ifndef _FLIP_FLUID_SOLVER_H_
#define _FLIP_FLUID_SOLVER_H_

#include <memory>

#include "fluid_solver.h"
#include "graphics_linear_mem.h"
#include "third_party/glm/vec3.hpp"

class GraphicsMemPiece;
class GraphicsVolume;
class GraphicsVolume3;
class PoissonCore;
class PoissonSolver;
class FlipFluidSolver : public FluidSolver
{
public:
    FlipFluidSolver();
    virtual ~FlipFluidSolver();

    virtual void Impulse(GraphicsVolume* density, float splat_radius,
                         const glm::vec3& impulse_position,
                         const glm::vec3& hotspot, float impulse_density,
                         float impulse_temperature) override;
    virtual bool Initialize(GraphicsLib graphics_lib, int width, int height,
                            int depth) override;
    virtual void Reset() override;
    virtual void SetDiagnosis(int diagnosis) override;
    virtual void SetPressureSolver(PoissonSolver* solver) override;
    virtual void Solve(GraphicsVolume* density, float delta_time) override;

private:
    struct FlipParticles;

    static bool InitParticles(FlipParticles* particles, GraphicsLib lib,
                              int cell_count);

    void AdvectTemperature(float delta_time);
    void ApplyBuoyancy(const GraphicsVolume& density, float delta_time);
    void ComputeDivergence(std::shared_ptr<GraphicsVolume> divergence);
    void MoveParticles(GraphicsVolume* density, float delta_time);
    void SolvePressure(std::shared_ptr<GraphicsVolume> pressure,
                       std::shared_ptr<GraphicsVolume> divergence,
                       int num_iterations);
    void SubtractGradient(std::shared_ptr<GraphicsVolume> pressure);

    GraphicsLib graphics_lib_;
    glm::ivec3 grid_size_;
    PoissonSolver* pressure_solver_;
    int diagnosis_;

    std::shared_ptr<GraphicsVolume3> velocity_;
    std::shared_ptr<GraphicsVolume3> velocity_prev_;
    std::shared_ptr<GraphicsVolume> temperature_;
    std::shared_ptr<GraphicsVolume> general1a_;
    std::shared_ptr<GraphicsVolume> general1b_;
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;

    std::unique_ptr<FlipParticles> particles_;
    std::shared_ptr<GraphicsLinearMemU16> aux_;

    int frame_;
    int num_active_particles_;
};

#endif // _FLIP_FLUID_SOLVER_H_