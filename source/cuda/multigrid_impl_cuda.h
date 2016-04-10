#ifndef _MULTIGRID_IMPL_CUDA_H_
#define _MULTIGRID_IMPL_CUDA_H_

#include <memory>

struct cudaArray;
namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class CudaVolume;
class GraphicsResource;
class MultigridImplCuda
{
public:
    MultigridImplCuda();
    ~MultigridImplCuda();

    void ComputeResidualPackedPure(cudaArray* dest_array,
                                   cudaArray* source_array,
                                   float inverse_h_square,
                                   const Vectormath::Aos::Vector3& volume_size);
    void ProlongatePackedPure(cudaArray* dest, cudaArray* coarse,
                              cudaArray* fine,
                              const Vectormath::Aos::Vector3& volume_size);
    void RelaxWithZeroGuessPackedPure(
        cudaArray* dest_array, cudaArray* source_array,
        float alpha_omega_over_beta, float one_minus_omega,
        float minus_h_square, float omega_times_inverse_beta,
        const Vectormath::Aos::Vector3& volume_size);
    void RestrictPackedPure(cudaArray* dest_array, cudaArray* source_array,
                            const Vectormath::Aos::Vector3& volume_size);
    void RestrictResidualPackedPure(
        cudaArray* dest_array, cudaArray* source_array,
        const Vectormath::Aos::Vector3& volume_size);
};

#endif // _MULTIGRID_IMPL_CUDA_H_