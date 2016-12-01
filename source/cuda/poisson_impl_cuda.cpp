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

#include "poisson_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>
#include <driver_types.h>

#include "graphics_resource.h"
#include "kernel_launcher.h"
#include "third_party/glm/vec3.hpp"

namespace
{
uint3 FromGlmVector(const glm::ivec3& v)
{
    return make_uint3(v.x, v.y, v.z);
}
} // Anonymous namespace.

PoissonImplCuda::PoissonImplCuda(BlockArrangement* ba, AuxBufferManager* bm)
    : ba_(ba)
    , bm_(bm)
    , cell_size_(0.15f)
    , outflow_(false)
{
}

PoissonImplCuda::~PoissonImplCuda()
{
}

void PoissonImplCuda::ComputeResidual(cudaArray* r, cudaArray* u,
                                      cudaArray* b,
                                      const glm::ivec3& volume_size)
{
    kern_launcher::ComputeResidual(r, u, b, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::Prolongate(cudaArray* fine, cudaArray* coarse,
                                 const glm::ivec3& volume_size)
{
    kern_launcher::Prolongate(fine, coarse, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::ProlongateError(cudaArray* fine, cudaArray* coarse,
                                      const glm::ivec3& volume_size)
{
    kern_launcher::ProlongateError(fine, coarse, FromGlmVector(volume_size),
                                   ba_);
}

void PoissonImplCuda::RelaxWithZeroGuess(cudaArray* u, cudaArray* b,
                                         const glm::ivec3& volume_size)
{
    kern_launcher::RelaxWithZeroGuess(u, b, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::Restrict(cudaArray* coarse, cudaArray* fine,
                               const glm::ivec3& volume_size)
{
    kern_launcher::Restrict(coarse, fine, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::ApplyStencil(cudaArray* aux, cudaArray* search,
                                   const glm::ivec3& volume_size)
{
    kern_launcher::ApplyStencil(aux, search, outflow_,
                                FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::ComputeAlpha(const MemPiece& alpha, const MemPiece& rho,
                                   cudaArray* aux, cudaArray* search,
                                   const glm::ivec3& volume_size)
{
    kern_launcher::ComputeAlpha(alpha, rho, aux, search,
                                FromGlmVector(volume_size), ba_, bm_);
}

void PoissonImplCuda::ComputeRho(const MemPiece& rho, cudaArray* search,
                                 cudaArray* residual,
                                 const glm::ivec3& volume_size)
{
    kern_launcher::ComputeRho(rho, search, residual, FromGlmVector(volume_size),
                              ba_, bm_);
}

void PoissonImplCuda::ComputeRhoAndBeta(const MemPiece& beta,
                                        const MemPiece& rho_new,
                                        const MemPiece& rho, cudaArray* aux,
                                        cudaArray* residual,
                                        const glm::ivec3& volume_size)
{
    kern_launcher::ComputeRhoAndBeta(beta, rho_new, rho, aux, residual,
                                     FromGlmVector(volume_size), ba_, bm_);
}

void PoissonImplCuda::ScaledAdd(cudaArray* dest, cudaArray* v0, cudaArray* v1,
                                const MemPiece& coef, float sign,
                                const glm::ivec3& volume_size)
{
    kern_launcher::ScaledAdd(dest, v0, v1, coef, sign,
                             FromGlmVector(volume_size), ba_);
}
