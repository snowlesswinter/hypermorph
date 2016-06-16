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
    , rho_()
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

    rho_ = core_->CreateMemPiece(sizeof(float));
    if (!rho_)
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
    std::shared_ptr<GraphicsVolume> r;

    core_->ComputeResidual(*r, *u, *b, cell_size);

    preconditioner_->Solve(aux_, r, cell_size, 1);
    // d <- aux

    core_->ComputeRho(*rho_, *r, *aux_);
    for (int i = 0; i < iteration_times; i++) {
        // Apply matrix A to |aux|
        // d <- A(aux)

        // alpha <- rho / dot(aux, r)
        // r <- r - alpha * aux

        preconditioner_->Solve(aux_, r, cell_size, 1);

        // rho2 <- dot(aux, r)
        // beta <- rho2 / rho
        // swap(rho2, rho)

        // u <- u + alpha * d
        // d <- aux + beta * d
    }
}
