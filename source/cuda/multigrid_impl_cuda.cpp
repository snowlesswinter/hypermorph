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
#include "third_party/glm/vec3.hpp"

extern void LaunchComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                                  float cell_size, uint3 volume_size,
                                  BlockArrangement* ba);
extern void LaunchProlongate(cudaArray* fine, cudaArray* coarse,
                             uint3 volume_size_fine, BlockArrangement* ba);
extern void LaunchProlongateError(cudaArray* fine, cudaArray* coarse,
                                  uint3 volume_size_fine, BlockArrangement* ba);
extern void LaunchRelaxWithZeroGuess(cudaArray* u, cudaArray* b,
                                     float cell_size, uint3 volume_size,
                                     BlockArrangement* ba);
extern void LaunchRestrict(cudaArray* coarse, cudaArray* fine,
                           uint3 volume_size);

namespace
{
uint3 FromGlmVector(const glm::ivec3& v)
{
    return make_uint3(v.x, v.y, v.z);
}
} // Anonymous namespace.

MultigridImplCuda::MultigridImplCuda(BlockArrangement* ba)
    :ba_(ba)
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
    LaunchRestrict(coarse, fine, FromGlmVector(volume_size));
}
