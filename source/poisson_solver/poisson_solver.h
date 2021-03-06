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

#ifndef _POISSON_SOLVER_H_
#define _POISSON_SOLVER_H_

#include <memory>
#include <vector>

class GraphicsVolume;
class PoissonSolver
{
public:
    PoissonSolver();
    virtual ~PoissonSolver();

    // The convergence rates of Poisson solvers are sensitive to the precision
    // of data presentation. Experiment results indicate that, fp16 data type
    // always leads to very slow convergence, and the solvers would sometimes
    // stop converging further since a small iteration number.
    //
    // When I turn to fp32, the convergence rate start to raise as expected,
    // and MGPCG solver is then able to achieve a better result if I increase
    // the iteration times.
    //
    // However, in practice, the high precision solution seems not to bring
    // much more visual details compares to its expensive computation. Hence,
    // I still stick to fp16 for performance consideration, and switch to
    // higher precision mode for algorithm verification.

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width, int minimum_grid_width) = 0;
    virtual void SetAuxiliaryVolumes(
        const std::vector<std::shared_ptr<GraphicsVolume>>& volumes) = 0;
    virtual void SetDiagnosis(bool diagnosis) = 0;
    virtual void SetNumOfIterations(int num_iterations, int nested_solver) = 0;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u,
                       std::shared_ptr<GraphicsVolume> b) = 0;
};

#endif // _POISSON_SOLVER_H_