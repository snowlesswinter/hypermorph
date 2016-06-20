#include "multigrid_impl_cuda.h"

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
{
}

PoissonImplCuda::~PoissonImplCuda()
{
}

void PoissonImplCuda::ComputeResidual(cudaArray* r, cudaArray* u,
                                      cudaArray* b, float cell_size,
                                      const glm::ivec3& volume_size)
{
    LaunchComputeResidual(r, u, b, cell_size, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::Prolongate(cudaArray* fine, cudaArray* coarse,
                                 const glm::ivec3& volume_size)
{
    LaunchProlongate(fine, coarse, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::ProlongateError(cudaArray* fine, cudaArray* coarse,
                                      const glm::ivec3& volume_size)
{
    LaunchProlongateError(fine, coarse, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::RelaxWithZeroGuess(cudaArray* u, cudaArray* b,
                                         float cell_size,
                                         const glm::ivec3& volume_size)
{
    LaunchRelaxWithZeroGuess(u, b, cell_size, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::Restrict(cudaArray* coarse, cudaArray* fine,
                               const glm::ivec3& volume_size)
{
    LaunchRestrict(coarse, fine, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::ApplyStencil(cudaArray* aux, cudaArray* search,
                                   float cell_size,
                                   const glm::ivec3& volume_size)
{
    LaunchApplyStencil(aux, search, cell_size, FromGlmVector(volume_size), ba_);
}

void PoissonImplCuda::ComputeAlpha(float* alpha, float* rho, cudaArray* aux,
                                   cudaArray* search,
                                   const glm::ivec3& volume_size)
{
    LaunchComputeAlpha(alpha, rho, aux, search, FromGlmVector(volume_size), ba_,
                       bm_);
}

void PoissonImplCuda::ComputeRho(float* rho, cudaArray* search,
                                 cudaArray* residual,
                                 const glm::ivec3& volume_size)
{
    LaunchComputeRho(rho, search, residual, FromGlmVector(volume_size), ba_,
                     bm_);
}

void PoissonImplCuda::ComputeRhoAndBeta(float* beta, float* rho_new,
                                        float* rho, cudaArray* aux,
                                        cudaArray* residual,
                                        const glm::ivec3& volume_size)
{
    LaunchComputeRhoAndBeta(beta, rho_new, rho, aux, residual,
                            FromGlmVector(volume_size), ba_, bm_);
}

void PoissonImplCuda::UpdateVector(cudaArray* dest, cudaArray* v0,
                                   cudaArray* v1, float* coef, float sign,
                                   const glm::ivec3& volume_size)
{
    LaunchUpdateVector(dest, v0, v1, coef, sign, FromGlmVector(volume_size),
                       ba_);
}
