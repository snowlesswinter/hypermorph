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

#ifndef _POISSON_CORE_H_
#define _POISSON_CORE_H_

#include <memory>

class GraphicsMemPiece;
class GraphicsVolume;
class GraphicsVolume3;
class PoissonCore
{
public:
    PoissonCore();
    virtual ~PoissonCore();

    virtual std::shared_ptr<GraphicsMemPiece> CreateMemPiece(int size) = 0;
    virtual std::shared_ptr<GraphicsVolume> CreateVolume(int width, int height,
                                                         int depth,
                                                         int num_of_components,
                                                         int byte_width) = 0;
    virtual std::shared_ptr<GraphicsVolume3> CreateVolumeGroup(
        int width, int height, int depth, int num_of_components,
        int byte_width) = 0;

    // Multigrid.
    virtual void ComputeResidual(const GraphicsVolume& r,
                                 const GraphicsVolume& u,
                                 const GraphicsVolume& b) = 0;
    virtual void Prolongate(const GraphicsVolume& fine,
                            const GraphicsVolume& coarse) = 0;
    virtual void ProlongateError(const GraphicsVolume& fine,
                                 const GraphicsVolume& coarse) = 0;
    virtual void Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                       int num_of_iterations) = 0;
    virtual void RelaxWithZeroGuess(const GraphicsVolume& u,
                                    const GraphicsVolume& b) = 0;
    virtual void Restrict(const GraphicsVolume& coarse,
                          const GraphicsVolume& fine) = 0;

    // Conjugate gradient.
    virtual void ApplyStencil(const GraphicsVolume& aux,
                              const GraphicsVolume& search) = 0;
    virtual void ComputeAlpha(const GraphicsMemPiece& alpha,
                              const GraphicsMemPiece& rho,
                              const GraphicsVolume& aux,
                              const GraphicsVolume& search) = 0;
    virtual void ComputeRho(const GraphicsMemPiece& rho,
                            const GraphicsVolume& search,
                            const GraphicsVolume& residual) = 0;
    virtual void ComputeRhoAndBeta(const GraphicsMemPiece& beta,
                                   const GraphicsMemPiece& rho_new,
                                   const GraphicsMemPiece& rho,
                                   const GraphicsVolume& aux,
                                   const GraphicsVolume& residual) = 0;
    virtual void ScaledAdd(const GraphicsVolume& dest, const GraphicsVolume& v0,
                           const GraphicsVolume& v1,
                           const GraphicsMemPiece& coef, float sign) = 0;
    virtual void ScaleVector(const GraphicsVolume& dest,
                             const GraphicsVolume& v,
                             const GraphicsMemPiece& coef, float sign) = 0;
};

#endif // _POISSON_CORE_H_