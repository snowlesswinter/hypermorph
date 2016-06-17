#include "stdafx.h"
#include "preconditioned_conjugate_gradient.h"

#include <cassert>

#include "graphics_mem_piece.h"
#include "graphics_volume.h"
#include "multigrid_core.h"
#include "multigrid_poisson_solver.h"

PreconditionedConjugateGradient::PreconditionedConjugateGradient(
        MultigridCore* core)
    : core_(core)
    , preconditioner_()
    , alpha_()
    , beta_()
    , rho_()
    , rho_new_()
    , residual_()
    , aux_()
    , search_()
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

    residual_ = core_->CreateVolume(width, height, depth, 1, byte_width);
    if (!residual_)
        return false;

    aux_ = core_->CreateVolume(width, height, depth, 1, byte_width);
    if (!aux_)
        return false;

    search_ = core_->CreateVolume(width, height, depth, 1, byte_width);
    if (!search_)
        return false;

    return false;
}

void PreconditionedConjugateGradient::Solve(std::shared_ptr<GraphicsVolume> u,
                                            std::shared_ptr<GraphicsVolume> b,
                                            float cell_size,
                                            int iteration_times)
{
    core_->ComputeResidual(*residual_, *u, *b, cell_size);

    preconditioner_->Solve(aux_, residual_, cell_size, 1);
    // d <- aux

    core_->ComputeRho(*rho_, *residual_, *aux_);
    for (int i = 0; i < iteration_times; i++) {
        core_->ApplyStencil(*aux_, *search_, cell_size);

        core_->ComputeAlpha(*alpha_, *rho_, *aux_, *search_);
        core_->UpdateVector(*residual_, *aux_, *alpha_, -1.0f);

        preconditioner_->Solve(aux_, residual_, cell_size, 1);

        core_->ComputeRhoAndBeta(*beta_, *rho_new_, *rho_, *aux_, *residual_);
        std::swap(rho_new_, rho_);

        core_->UpdateVector(*u, *search_, *alpha_, 1.0f);
        core_->UpdateVector(*search_, *search_, *beta_, 1.0f);
    }
}
