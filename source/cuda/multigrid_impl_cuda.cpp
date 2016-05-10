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

extern void LaunchComputeResidualPackedPure(cudaArray* dest_array,
                                            cudaArray* source_array,
                                            float inverse_h_square,
                                            uint3 volume_size,
                                            BlockArrangement* ba);
extern void LaunchProlongatePackedPure(cudaArray* dest_array,
                                       cudaArray* coarse_array,
                                       cudaArray* fine_array, float overlay,
                                       uint3 volume_size_fine,
                                       BlockArrangement* ba);
extern void LaunchRelaxWithZeroGuessPackedPure(cudaArray* dest_array,
                                               cudaArray* source_array,
                                               float alpha_omega_over_beta,
                                               float one_minus_omega,
                                               float minus_h_square,
                                               float omega_times_inverse_beta,
                                               uint3 volume_size);
extern void LaunchRestrictPackedPure(cudaArray* dest_array,
                                     cudaArray* source_array, uint3 volume_size,
                                     BlockArrangement* ba);
extern void LaunchRestrictResidualPackedPure(cudaArray* dest_array,
                                             cudaArray* source_array,
                                             uint3 volume_size);

namespace
{
uint3 FromVmathVector(const glm::ivec3& v)
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

void MultigridImplCuda::ComputeResidualPackedPure(cudaArray* dest_array,
                                                  cudaArray* source_array,
                                                  float inverse_h_square,
                                                  const glm::ivec3& volume_size)
{
    LaunchComputeResidualPackedPure(dest_array, source_array, inverse_h_square,
                                    FromVmathVector(volume_size), ba_);
}

void MultigridImplCuda::ProlongatePackedPure(cudaArray* dest, cudaArray* coarse,
                                             cudaArray* fine, float overlay,
                                             const glm::ivec3& volume_size)
{
    LaunchProlongatePackedPure(dest, coarse, fine, overlay,
                               FromVmathVector(volume_size), ba_);
}

void MultigridImplCuda::RelaxWithZeroGuessPackedPure(
    cudaArray* dest_array, cudaArray* source_array, float alpha_omega_over_beta,
    float one_minus_omega, float minus_h_square, float omega_times_inverse_beta,
    const glm::ivec3& volume_size)
{
    LaunchRelaxWithZeroGuessPackedPure(dest_array, source_array,
                                       alpha_omega_over_beta, one_minus_omega,
                                       minus_h_square, omega_times_inverse_beta,
                                       FromVmathVector(volume_size));
}

void MultigridImplCuda::RestrictPackedPure(cudaArray* dest_array,
                                           cudaArray* source_array,
                                           const glm::ivec3& volume_size)
{
    LaunchRestrictPackedPure(dest_array, source_array,
                             FromVmathVector(volume_size), ba_);
}

void MultigridImplCuda::RestrictResidualPackedPure(
    cudaArray* dest_array, cudaArray* source_array,
    const glm::ivec3& volume_size)
{
    LaunchRestrictResidualPackedPure(dest_array, source_array,
                                     FromVmathVector(volume_size));
}
