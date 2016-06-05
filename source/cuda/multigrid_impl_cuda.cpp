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

extern void LaunchComputeResidualPacked(cudaArray* dest_array,
                                        cudaArray* source_array,
                                        float inverse_h_square,
                                        uint3 volume_size,
                                        BlockArrangement* ba);
extern void LaunchComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                                  float cell_size, uint3 volume_size,
                                  BlockArrangement* ba);
extern void LaunchProlongate(cudaArray* fine, cudaArray* coarse,
                             uint3 volume_size_fine, BlockArrangement* ba);
extern void LaunchProlongatePacked(cudaArray* dest_array,
                                   cudaArray* coarse_array,
                                   cudaArray* fine_array, float overlay,
                                   uint3 volume_size_fine,
                                   BlockArrangement* ba);
extern void LaunchRelaxWithZeroGuess(cudaArray* u, cudaArray* b,
                                     float cell_size, uint3 volume_size,
                                     BlockArrangement* ba);
extern void LaunchRelaxWithZeroGuessPacked(cudaArray* dest_array,
                                           cudaArray* source_array,
                                           float alpha_omega_over_beta,
                                           float one_minus_omega,
                                           float minus_h_square,
                                           float omega_times_inverse_beta,
                                           uint3 volume_size);
extern void LaunchRestrict(cudaArray* coarse, cudaArray* fine,
                           uint3 volume_size);
extern void LaunchRestrictPacked(cudaArray* dest_array, cudaArray* source_array,
                                 uint3 volume_size, BlockArrangement* ba);
extern void LaunchRestrictResidualPacked(cudaArray* dest_array,
                                         cudaArray* source_array,
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

void MultigridImplCuda::ComputeResidualPacked(cudaArray* dest_array,
                                              cudaArray* source_array,
                                              float inverse_h_square,
                                              const glm::ivec3& volume_size)
{
    LaunchComputeResidualPacked(dest_array, source_array, inverse_h_square,
                                FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::Prolongate(cudaArray* fine, cudaArray* coarse,
                                   const glm::ivec3& volume_size)
{
    LaunchProlongate(fine, coarse, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::ProlongatePacked(cudaArray* dest, cudaArray* coarse,
                                         cudaArray* fine, float overlay,
                                         const glm::ivec3& volume_size)
{
    LaunchProlongatePacked(dest, coarse, fine, overlay,
                           FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::RelaxWithZeroGuess(cudaArray* u, cudaArray* b,
                                           float cell_size,
                                           const glm::ivec3& volume_size)
{
    LaunchRelaxWithZeroGuess(u, b, cell_size, FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::RelaxWithZeroGuessPacked(cudaArray* dest_array,
                                                 cudaArray* source_array,
                                                 float alpha_omega_over_beta,
                                                 float one_minus_omega,
                                                 float minus_h_square,
                                                 float omega_times_inverse_beta,
                                                 const glm::ivec3& volume_size)
{
    LaunchRelaxWithZeroGuessPacked(dest_array, source_array,
                                   alpha_omega_over_beta, one_minus_omega,
                                   minus_h_square, omega_times_inverse_beta,
                                   FromGlmVector(volume_size));
}

void MultigridImplCuda::Restrict(cudaArray* coarse, cudaArray* fine,
                                 const glm::ivec3& volume_size)
{
    LaunchRestrict(coarse, fine, FromGlmVector(volume_size));
}

void MultigridImplCuda::RestrictPacked(cudaArray* dest_array,
                                       cudaArray* source_array,
                                       const glm::ivec3& volume_size)
{
    LaunchRestrictPacked(dest_array, source_array,
                         FromGlmVector(volume_size), ba_);
}

void MultigridImplCuda::RestrictResidualPacked(cudaArray* dest_array,
                                               cudaArray* source_array,
                                               const glm::ivec3& volume_size)
{
    LaunchRestrictResidualPacked(dest_array, source_array,
                                 FromGlmVector(volume_size));
}
