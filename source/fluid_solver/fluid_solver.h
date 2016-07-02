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

#ifndef _FLUID_SOLVER_H_
#define _FLUID_SOLVER_H_

#include "graphics_lib_enum.h"
#include "third_party/glm/fwd.hpp"

class GraphicsVolume;
class PoissonSolver;
class FluidSolver
{
public:
    struct FluidProperties
    {
        float velocity_dissipation_;
        float density_dissipation_;
        float temperature_dissipation_;
        float weight_;
        float ambient_temperature_;
        float vorticity_confinement_;
        float buoyancy_coef_;
    };

    FluidSolver();
    virtual ~FluidSolver();

    virtual void Impulse(GraphicsVolume* density, float splat_radius,
                         const glm::vec3& impulse_position,
                         const glm::vec3& hotspot, float impulse_density,
                         float impulse_temperature) = 0;
    virtual bool Initialize(GraphicsLib graphics_lib, int width, int height,
                            int depth) = 0;
    virtual void Reset() = 0;
    virtual void SetDiagnosis(int diagnosis) = 0;
    virtual void SetPressureSolver(PoissonSolver* solver) = 0;
    virtual void SetProperties(const FluidProperties& properties);
    virtual void Solve(GraphicsVolume* density, float delta_time) = 0;

protected:
    const FluidProperties& GetProperties() const { return properties_;  }

private:
    FluidProperties properties_;
};

#endif // _FLUID_SOLVER_H_