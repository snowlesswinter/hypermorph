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
#include "poisson_core_cuda.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "graphics_mem_piece.h"
#include "graphics_volume.h"
#include "graphics_volume_group.h"
#include "utility.h"

PoissonCoreCuda::PoissonCoreCuda()
    : PoissonCore()
{

}

PoissonCoreCuda::~PoissonCoreCuda()
{

}

std::shared_ptr<GraphicsMemPiece> PoissonCoreCuda::CreateMemPiece(int size)
{
    std::shared_ptr<GraphicsMemPiece> r =
        std::make_shared<GraphicsMemPiece>(GRAPHICS_LIB_CUDA);
    bool succeeded = r->Create(size);
    return succeeded ? r : std::shared_ptr<GraphicsMemPiece>();
}

std::shared_ptr<GraphicsVolume> PoissonCoreCuda::CreateVolume(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> r =
        std::make_shared<GraphicsVolume>(GRAPHICS_LIB_CUDA);
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);

    return succeeded ? r : std::shared_ptr<GraphicsVolume>();
}

std::shared_ptr<GraphicsVolume3> PoissonCoreCuda::CreateVolumeGroup(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume3> r(new GraphicsVolume3(GRAPHICS_LIB_CUDA));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);
    return succeeded ? r : std::shared_ptr<GraphicsVolume3>();
}

void PoissonCoreCuda::ComputeResidual(const GraphicsVolume& r,
                                      const GraphicsVolume& u,
                                      const GraphicsVolume& b)
{
    CudaMain::Instance()->ComputeResidual(r.cuda_volume(), u.cuda_volume(),
                                          b.cuda_volume());
}

void PoissonCoreCuda::Prolongate(const GraphicsVolume& fine,
                                 const GraphicsVolume& coarse)
{
    CudaMain::Instance()->Prolongate(fine.cuda_volume(), coarse.cuda_volume());
}

void PoissonCoreCuda::ProlongateError(const GraphicsVolume& fine,
                                      const GraphicsVolume& coarse)
{
    CudaMain::Instance()->ProlongateError(fine.cuda_volume(),
                                          coarse.cuda_volume());
}

void PoissonCoreCuda::Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                            int num_of_iterations)
{
    CudaMain::Instance()->Relax(u.cuda_volume(), u.cuda_volume(),
                                b.cuda_volume(), num_of_iterations);
}

void PoissonCoreCuda::RelaxWithZeroGuess(const GraphicsVolume& u,
                                         const GraphicsVolume& b)
{
    CudaMain::Instance()->RelaxWithZeroGuess(u.cuda_volume(), b.cuda_volume());
}

void PoissonCoreCuda::Restrict(const GraphicsVolume& coarse,
                               const GraphicsVolume& fine)
{
    CudaMain::Instance()->Restrict(coarse.cuda_volume(), fine.cuda_volume());
}

void PoissonCoreCuda::ApplyStencil(const GraphicsVolume& aux,
                                   const GraphicsVolume& search)
{
    CudaMain::Instance()->ApplyStencil(aux.cuda_volume(), search.cuda_volume());
}

void PoissonCoreCuda::ComputeAlpha(const GraphicsMemPiece& alpha,
                                   const GraphicsMemPiece& rho,
                                   const GraphicsVolume& aux,
                                   const GraphicsVolume& search)
{
    CudaMain::Instance()->ComputeAlpha(alpha.cuda_mem_piece(),
                                       rho.cuda_mem_piece(), aux.cuda_volume(),
                                       search.cuda_volume());
}

void PoissonCoreCuda::ComputeRho(const GraphicsMemPiece& rho,
                                 const GraphicsVolume& search,
                                 const GraphicsVolume& residual)
{
    CudaMain::Instance()->ComputeRho(rho.cuda_mem_piece(), search.cuda_volume(),
                                     residual.cuda_volume());
}

void PoissonCoreCuda::ComputeRhoAndBeta(const GraphicsMemPiece& beta,
                                        const GraphicsMemPiece& rho_new,
                                        const GraphicsMemPiece& rho,
                                        const GraphicsVolume& aux,
                                        const GraphicsVolume& residual)
{
    CudaMain::Instance()->ComputeRhoAndBeta(beta.cuda_mem_piece(),
                                            rho_new.cuda_mem_piece(),
                                            rho.cuda_mem_piece(),
                                            aux.cuda_volume(),
                                            residual.cuda_volume());
}

void PoissonCoreCuda::ScaledAdd(const GraphicsVolume& dest,
                                const GraphicsVolume& v0,
                                const GraphicsVolume& v1,
                                const GraphicsMemPiece& coef, float sign)
{
    CudaMain::Instance()->ScaledAdd(dest.cuda_volume(), v0.cuda_volume(),
                                       v1.cuda_volume(), coef.cuda_mem_piece(),
                                       sign);
}

void PoissonCoreCuda::ScaleVector(const GraphicsVolume& dest,
                                  const GraphicsVolume& v,
                                  const GraphicsMemPiece& coef,
                                  float sign)
{
    CudaMain::Instance()->ScaleVector(dest.cuda_volume(), v.cuda_volume(),
                                      coef.cuda_mem_piece(), sign);
}
