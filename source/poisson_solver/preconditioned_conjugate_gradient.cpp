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

#include "stdafx.h"
#include "preconditioned_conjugate_gradient.h"

#include <cassert>

#include "graphics_mem_piece.h"
#include "graphics_volume.h"
#include "multigrid_poisson_solver.h"
#include "poisson_core.h"

PreconditionedConjugateGradient::PreconditionedConjugateGradient(
        PoissonCore* core)
    : core_(core)
    , preconditioner_(new MultigridPoissonSolver(core))
    , alpha_()
    , beta_()
    , rho_()
    , rho_new_()
    , residual_()
    , aux_()
    , search_()
    , num_nested_iterations_(2)
    , diagnosis_(false)
{

}

PreconditionedConjugateGradient::~PreconditionedConjugateGradient()
{

}

bool PreconditionedConjugateGradient::Initialize(int width, int height,
                                                 int depth, int byte_width,
                                                 int minimum_grid_width)
{
    if (!preconditioner_->Initialize(width, height, depth, byte_width,
                                     minimum_grid_width))
        return false;

    alpha_ = core_->CreateMemPiece(sizeof(float));
    if (!alpha_)
        return false;

    beta_ = core_->CreateMemPiece(sizeof(float));
    if (!beta_)
        return false;

    rho_ = core_->CreateMemPiece(sizeof(float));
    if (!rho_)
        return false;

    rho_new_ = core_->CreateMemPiece(sizeof(float));
    if (!rho_new_)
        return false;

    if (!aux_) {
        aux_ = core_->CreateVolume(width, height, depth, 1, byte_width);
        if (!aux_)
            return false;
    }

    if (!search_) {
        search_ = core_->CreateVolume(width, height, depth, 1, byte_width);
        if (!search_)
            return false;
    }

    return true;
}

void PreconditionedConjugateGradient::SetAuxiliaryVolumes(
    const std::vector<std::shared_ptr<GraphicsVolume>>& volumes)
{
    if (volumes.size() >= 1)
        aux_ = volumes[0];

    if (volumes.size() >= 2)
        search_ = volumes[1];
}

void PreconditionedConjugateGradient::SetDiagnosis(bool diagnosis)
{
    diagnosis_ = diagnosis;
}

void PreconditionedConjugateGradient::SetNestedSolverIterations(
    int num_iterations)
{
    num_nested_iterations_ = num_iterations;
}

void PreconditionedConjugateGradient::Solve(std::shared_ptr<GraphicsVolume> u,
                                            std::shared_ptr<GraphicsVolume> b,
                                            int iteration_times)
{
    bool initialized = false;
    std::shared_ptr<GraphicsVolume> r = b;

    // |residual_| is actually not necessary in solving the pressure. It is
    // diagnosing that require an extra buffer to store the temporary data so
    // that |b| can be used to compute residual later.
    if (diagnosis_ && iteration_times > 1) {
        if (!residual_) {
            residual_ = core_->CreateVolume(b->GetWidth(), b->GetHeight(),
                                            b->GetDepth(), 1,
                                            b->GetByteWidth());
            if (!residual_)
                return;
        }

        u->Clear();

        // Copy |b| to |residual_|.
        core_->ComputeResidual(*residual_, *u, *b);
        r = residual_;
        initialized = true;
    }

    preconditioner_->set_num_finest_level_iteration_per_pass(
        num_nested_iterations_);
    preconditioner_->Solve(search_, r, 1);
    core_->ComputeRho(*rho_, *search_, *r);
    for (int i = 0; i < iteration_times - 1; i++) {
        core_->ApplyStencil(*aux_, *search_);

        core_->ComputeAlpha(*alpha_, *rho_, *aux_, *search_);
        core_->ScaledAdd(*r, *r, *aux_, *alpha_, -1.0f);

        preconditioner_->Solve(aux_, r, 1);

        core_->ComputeRhoAndBeta(*beta_, *rho_new_, *rho_, *aux_, *r);
        std::swap(rho_new_, rho_);

        UpdateU(*u, *search_, *alpha_, &initialized);
        core_->ScaledAdd(*search_, *aux_, *search_, *beta_, 1.0f);
    }

    core_->ApplyStencil(*aux_, *search_);
    core_->ComputeAlpha(*alpha_, *rho_, *aux_, *search_);
    UpdateU(*u, *search_, *alpha_, &initialized);
}

void PreconditionedConjugateGradient::UpdateU(const GraphicsVolume& u,
                                              const GraphicsVolume& search,
                                              const GraphicsMemPiece& alpha,
                                              bool* initialized)
{
    if (*initialized)
        core_->ScaledAdd(u, u, search, alpha, 1.0f);
    else
        core_->ScaleVector(u, search, alpha, 1.0f);

    *initialized = true;
}
