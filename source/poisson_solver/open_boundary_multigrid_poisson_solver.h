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

#ifndef _OPEN_BOUNDARY_MULTIGRID_POISSON_SOLVER_H_
#define _OPEN_BOUNDARY_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "graphics_volume_group.h"

class PoissonCore;
class OpenBoundaryMultigridPoissonSolver
{
public:
    explicit OpenBoundaryMultigridPoissonSolver(PoissonCore* core);
    virtual ~OpenBoundaryMultigridPoissonSolver();

    bool Initialize(int width, int height, int depth,
                    int byte_width);
    void Solve(std::shared_ptr<GraphicsVolume> u,
               std::shared_ptr<GraphicsVolume> b);

    void set_num_finest_level_iteration_per_pass(int n) {
        num_finest_level_iteration_per_pass_ = n;
    }

private:
    void Relax(std::shared_ptr<GraphicsVolume> u,
               std::shared_ptr<GraphicsVolume> b, int times);

    PoissonCore* core_;
    std::vector<std::shared_ptr<GraphicsVolume3>> volume_resource_;
    std::shared_ptr<GraphicsVolume> residual_volume_;
    int num_finest_level_iteration_per_pass_;
};

#endif // _OPEN_BOUNDARY_MULTIGRID_POISSON_SOLVER_H_