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

MultigridImplCuda::MultigridImplCuda(BlockArrangement* ba, AuxBufferManager* bm)
    : ba_(ba)
    , bm_(bm)
{
}

MultigridImplCuda::~MultigridImplCuda()
{
}

void MultigridImplCuda::ComputeResidual(cudaArray* r, cudaArray* u,
                                        cudaArray* b, float cell_size,
                                        const glm::ivec3& volume_size)
{
    LaunchComputeResidual(r, u, b, cell_size, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::Prolongate(cudaArray* fine, cudaArray* coarse,
                                   const glm::ivec3& volume_size)
{
    LaunchProlongate(fine, coarse, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::ProlongateError(cudaArray* fine, cudaArray* coarse,
                                        const glm::ivec3& volume_size)
{
    LaunchProlongateError(fine, coarse, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::RelaxWithZeroGuess(cudaArray* u, cudaArray* b,
                                           float cell_size,
                                           const glm::ivec3& volume_size)
{
    LaunchRelaxWithZeroGuess(u, b, cell_size, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::Restrict(cudaArray* coarse, cudaArray* fine,
                                 const glm::ivec3& volume_size)
{
    LaunchRestrict(coarse, fine, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::ApplyStencil(cudaArray* aux, cudaArray* search,
                                     float cell_size,
                                     const glm::ivec3& volume_size)
{
    LaunchApplyStencil(aux, search, cell_size, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::ComputeAlpha(float* alpha, float* rho, cudaArray* aux,
                                     cudaArray* search,
                                     const glm::ivec3& volume_size)
{
    LaunchComputeAlpha(alpha, rho, aux, search, FromGlmVector(volume_size), ba_,
                       bm_);
}

void MultigridImplCuda::ComputeRho(float* rho, cudaArray* aux, cudaArray* r,
                                   const glm::ivec3& volume_size)
{
    LaunchComputeDotProductOfVectors(rho, aux, r, FromGlmVector(volume_size),
                                     ba_, bm_);
}

void MultigridImplCuda::ComputeRhoAndBeta(float* beta, float* rho_new,
                                          float* rho, cudaArray* aux,
                                          cudaArray* residual,
                                          const glm::ivec3& volume_size)
{
    LaunchComputeRhoAndBeta(beta, rho_new, rho, aux, residual,
                            FromGlmVector(volume_size), ba_, bm_);
}

void MultigridImplCuda::UpdateVector(cudaArray* dest, cudaArray* v,
                                     float* alpha, float sign,
                                     const glm::ivec3& volume_size)
{
    LaunchUpdateVector(dest, v, alpha, sign, FromGlmVector(volume_size), ba_);
}
