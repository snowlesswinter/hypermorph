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

#ifndef _POISSON_IMPL_CUDA_H_
#define _POISSON_IMPL_CUDA_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

struct cudaArray;
class AuxBufferManager;
class BlockArrangement;
class CudaVolume;
class GraphicsResource;
class PoissonImplCuda
{
public:
    PoissonImplCuda(BlockArrangement* ba, AuxBufferManager* bm);
    ~PoissonImplCuda();

    // Multigrid.
    void ComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                         float cell_size, const glm::ivec3& volume_size);
    void Prolongate(cudaArray* fine, cudaArray* coarse,
                    const glm::ivec3& volume_size);
    void ProlongateError(cudaArray* fine, cudaArray* coarse,
                         const glm::ivec3& volume_size);
    void RelaxWithZeroGuess(cudaArray* u, cudaArray* b, float cell_size,
                            const glm::ivec3& volume_size);
    void Restrict(cudaArray* coarse, cudaArray* fine,
                  const glm::ivec3& volume_size);

    // Conjugate gradient.
    void ApplyStencil(cudaArray* aux, cudaArray* search, float cell_size,
                      const glm::ivec3& volume_size);
    void ComputeAlpha(float* alpha, float* rho, cudaArray* aux,
                      cudaArray* search, const glm::ivec3& volume_size);
    void ComputeRho(float* rho, cudaArray* search, cudaArray* residual,
                    const glm::ivec3& volume_size);
    void ComputeRhoAndBeta(float* beta, float* rho_new, float* rho,
                           cudaArray* aux, cudaArray* residual,
                           const glm::ivec3& volume_size);
    void UpdateVector(cudaArray* dest, cudaArray* v0, cudaArray* v1,
                      float* coef, float sign, const glm::ivec3& volume_size);

    void set_outflow(bool outflow) { outflow_ = outflow; }

private:
    BlockArrangement* ba_;
    AuxBufferManager* bm_;
    bool outflow_;
};

#endif // _POISSON_IMPL_CUDA_H_