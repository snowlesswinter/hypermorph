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

#ifndef _PRECONDITIONED_CONJUGATE_GRADIENT_H_
#define _PRECONDITIONED_CONJUGATE_GRADIENT_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GraphicsMemPiece;
class MultigridPoissonSolver;
class PoissonCore;
class PreconditionedConjugateGradient : public PoissonSolver
{
public:
    explicit PreconditionedConjugateGradient(PoissonCore* core);
    virtual ~PreconditionedConjugateGradient();

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width, int minimum_grid_width) override;
    virtual void SetAuxiliaryVolumes(
        const std::vector<std::shared_ptr<GraphicsVolume>>& volumes) override;
    virtual void SetDiagnosis(bool diagnosis) override;
    virtual void SetNestedSolverIterations(int num_iterations) override;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u,
                       std::shared_ptr<GraphicsVolume> b, float cell_size,
                       int iteration_times) override;

private:
    void UpdateU(const GraphicsVolume& u, const GraphicsVolume& search,
                 const GraphicsMemPiece& alpha, bool* initialized);

    PoissonCore* core_;
    std::unique_ptr<MultigridPoissonSolver> preconditioner_;
    std::shared_ptr<GraphicsMemPiece> alpha_;
    std::shared_ptr<GraphicsMemPiece> beta_;
    std::shared_ptr<GraphicsMemPiece> rho_;
    std::shared_ptr<GraphicsMemPiece> rho_new_;
    std::shared_ptr<GraphicsVolume> residual_;
    std::shared_ptr<GraphicsVolume> aux_;
    std::shared_ptr<GraphicsVolume> search_;
    int num_nested_iterations_;
    bool diagnosis_;
};

#endif // _PRECONDITIONED_CONJUGATE_GRADIENT_H_