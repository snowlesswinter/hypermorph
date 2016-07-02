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

#ifndef _FULL_MULTIGRID_POISSON_SOLVER_H_
#define _FULL_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class MultigridPoissonSolver;
class PoissonCore;
class FullMultigridPoissonSolver : public PoissonSolver
{
public:
    explicit FullMultigridPoissonSolver(PoissonCore* core);
    virtual ~FullMultigridPoissonSolver();

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width, int minimum_grid_width) override;
    virtual void SetAuxiliaryVolumes(
        const std::vector<std::shared_ptr<GraphicsVolume>>& volumes) override;
    virtual void SetDiagnosis(bool diagnosis) override;
    virtual void SetNestedSolverIterations(int num_iterations) override;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u,
                       std::shared_ptr<GraphicsVolume> b,
                       int iteration_times) override;

private:
    typedef std::pair<std::shared_ptr<GraphicsVolume>,
        std::shared_ptr<GraphicsVolume>> VolumePair;

    void Iterate(std::shared_ptr<GraphicsVolume> u,
                 std::shared_ptr<GraphicsVolume> b,
                 bool apply_initial_guess);

    PoissonCore* core_;
    std::unique_ptr<MultigridPoissonSolver> solver_;
    std::vector<VolumePair> volume_resource_;
    int num_nested_iterations_;
};

#endif // _FULL_MULTIGRID_POISSON_SOLVER_H_