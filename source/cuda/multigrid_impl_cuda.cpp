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
#include "../vmath.hpp"

extern void LaunchComputeResidualPackedPure(cudaArray* dest_array,
                                            cudaArray* source_array,
                                            float inverse_h_square,
                                            int3 volume_size);
extern void LaunchProlongatePacked(float4* dest_array, cudaArray* coarse_array,
                                   cudaArray* fine_array,
                                   int3 volume_size_fine);
extern void LaunchProlongatePackedPure(cudaArray* dest_array,
                                       cudaArray* coarse_array,
                                       cudaArray* fine_array,
                                       int3 volume_size_fine);
extern void LaunchRelaxWithZeroGuessPackedPure(cudaArray* dest_array,
                                               cudaArray* source_array,
                                               float alpha_omega_over_beta,
                                               float one_minus_omega,
                                               float minus_h_square,
                                               float omega_times_inverse_beta,
                                               int3 volume_size);
extern void LaunchRestrictPackedPure(cudaArray* dest_array,
                                     cudaArray* source_array, int3 volume_size);
extern void LaunchRestrictResidualPackedPure(cudaArray* dest_array,
                                             cudaArray* source_array,
                                             int3 volume_size);

namespace
{
int3 FromVmathVector(const vmath::Vector3& v)
{
    return make_int3(static_cast<int>(v.getX()), static_cast<int>(v.getY()),
                     static_cast<int>(v.getZ()));
}
} // Anonymous namespace.

MultigridImplCuda::MultigridImplCuda()
{
}

MultigridImplCuda::~MultigridImplCuda()
{
}

void MultigridImplCuda::ComputeResidualPackedPure(
    cudaArray* dest_array, cudaArray* source_array, float inverse_h_square,
    const vmath::Vector3& volume_size)
{
    LaunchComputeResidualPackedPure(dest_array, source_array, inverse_h_square,
                                    FromVmathVector(volume_size));
}

void MultigridImplCuda::ProlongatePacked(GraphicsResource* coarse,
                                         GraphicsResource* fine,
                                         GraphicsResource* out_pbo,
                                         const vmath::Vector3& volume_size_fine)
{
    cudaGraphicsResource_t res[] = {
        coarse->resource(), fine->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    float4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Coarse texture.
    cudaArray* coarse_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&coarse_array,
                                                   coarse->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Fine texture.
    cudaArray* fine_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&fine_array,
                                                   fine->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchProlongatePacked(dest_array, coarse_array, fine_array,
                           FromVmathVector(volume_size_fine));

    //     float* a = new float[128 * 128 * 128 * 4];
    //     result = cudaMemcpy(a, dest_array, 128 * 128 * 128 * 4 * 4,
    //                         cudaMemcpyDeviceToHost);
    //     assert(result == cudaSuccess);
    //     if (result != cudaSuccess)
    //         return;
    //
    //     double p = 0;
    //     double sum = 0;
    //     for (int i = 0; i < 128; i++) {
    //         for (int j = 0; j < 128; j++) {
    //             for (int k = 0; k < 128; k++) {
    //                 for (int n = 0; n < 4; n += 4) {
    //                     float* z = &a[i * 128 * 128 * 4 + j * 128 * 4 + k * 4 + n];
    //                     p = *z;
    //                     float z0 = *z;
    //                     float z1 = *(z + 1);
    //                     float z2 = *(z + 2);
    //                     if (z0 != k || z1 != j || z2 != i)
    //                         sum += p;
    //                 }
    //             }
    //         }
    //     }
    // 
    //     delete[] a;

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}



void MultigridImplCuda::ProlongatePackedPure(
    cudaArray* dest, cudaArray* coarse, cudaArray* fine,
    const vmath::Vector3& volume_size)
{
    LaunchProlongatePackedPure(dest, coarse, fine,
                               FromVmathVector(volume_size));
}

void MultigridImplCuda::RelaxWithZeroGuessPackedPure(
    cudaArray* dest_array, cudaArray* source_array, float alpha_omega_over_beta,
    float one_minus_omega, float minus_h_square, float omega_times_inverse_beta,
    const vmath::Vector3& volume_size)
{
    LaunchRelaxWithZeroGuessPackedPure(dest_array, source_array,
                                       alpha_omega_over_beta, one_minus_omega,
                                       minus_h_square, omega_times_inverse_beta,
                                       FromVmathVector(volume_size));
}

void MultigridImplCuda::RestrictPackedPure(cudaArray* dest_array,
                                           cudaArray* source_array,
                                           const vmath::Vector3& volume_size)
{
    LaunchRestrictPackedPure(dest_array, source_array,
                             FromVmathVector(volume_size));
}

void MultigridImplCuda::RestrictResidualPackedPure(
    cudaArray* dest_array, cudaArray* source_array,
    const vmath::Vector3& volume_size)
{
    LaunchRestrictResidualPackedPure(dest_array, source_array,
                                     FromVmathVector(volume_size));
}
