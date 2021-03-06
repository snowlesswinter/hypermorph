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

#ifndef _MULTIGRID_CORE_CUDA_H_
#define _MULTIGRID_CORE_CUDA_H_

#include <memory>

#include "poisson_core.h"

class GLProgram;
class PoissonCoreCuda : public PoissonCore
{
public:
    PoissonCoreCuda();
    virtual ~PoissonCoreCuda();

    virtual std::shared_ptr<GraphicsMemPiece> CreateMemPiece(int size) override;
    virtual std::shared_ptr<GraphicsVolume> CreateVolume(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;
    virtual std::shared_ptr<GraphicsVolume3> CreateVolumeGroup(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;

    // Multigrid.
    virtual void ComputeResidual(const GraphicsVolume& r,
                                 const GraphicsVolume& u,
                                 const GraphicsVolume& b) override;
    virtual void Prolongate(const GraphicsVolume& fine,
                            const GraphicsVolume& coarse) override;
    virtual void ProlongateError(const GraphicsVolume& fine,
                                 const GraphicsVolume& coarse) override;
    virtual void Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                       int num_of_iterations) override;
    virtual void RelaxWithZeroGuess(const GraphicsVolume& u,
                                    const GraphicsVolume& b) override;
    virtual void Restrict(const GraphicsVolume& coarse,
                          const GraphicsVolume& fine) override;

    // Conjugate gradient.
    virtual void ApplyStencil(const GraphicsVolume& aux,
                              const GraphicsVolume& search) override;
    virtual void ComputeAlpha(const GraphicsMemPiece& alpha,
                              const GraphicsMemPiece& rho,
                              const GraphicsVolume& aux,
                              const GraphicsVolume& search) override;
    virtual void ComputeRhoAndBeta(const GraphicsMemPiece& beta,
                                   const GraphicsMemPiece& rho_new,
                                   const GraphicsMemPiece& rho,
                                   const GraphicsVolume& aux,
                                   const GraphicsVolume& residual) override;
    virtual void ComputeRho(const GraphicsMemPiece& rho,
                            const GraphicsVolume& search,
                            const GraphicsVolume& residual) override;
    virtual void ScaledAdd(const GraphicsVolume& dest, const GraphicsVolume& v0,
                           const GraphicsVolume& v1,
                           const GraphicsMemPiece& coef, float sign) override;
    virtual void ScaleVector(const GraphicsVolume& dest,
                             const GraphicsVolume& v,
                             const GraphicsMemPiece& coef,
                             float sign) override;
};

#endif // _MULTIGRID_CORE_CUDA_H_