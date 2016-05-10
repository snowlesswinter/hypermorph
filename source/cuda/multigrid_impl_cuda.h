#ifndef _MULTIGRID_IMPL_CUDA_H_
#define _MULTIGRID_IMPL_CUDA_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

struct cudaArray;
class BlockArrangement;
class CudaVolume;
class GraphicsResource;
class MultigridImplCuda
{
public:
    explicit MultigridImplCuda(BlockArrangement* ba);
    ~MultigridImplCuda();

    void ComputeResidualPackedPure(cudaArray* dest_array,
                                   cudaArray* source_array,
                                   float inverse_h_square,
                                   const glm::ivec3& volume_size);
    void ProlongatePackedPure(cudaArray* dest, cudaArray* coarse,
                              cudaArray* fine, float overlay,
                              const glm::ivec3& volume_size);
    void RelaxWithZeroGuessPackedPure(
        cudaArray* dest_array, cudaArray* source_array,
        float alpha_omega_over_beta, float one_minus_omega,
        float minus_h_square, float omega_times_inverse_beta,
        const glm::ivec3& volume_size);
    void RestrictPackedPure(cudaArray* dest_array, cudaArray* source_array,
                            const glm::ivec3& volume_size);
    void RestrictResidualPackedPure(cudaArray* dest_array,
                                    cudaArray* source_array,
                                    const glm::ivec3& volume_size);

private:
    BlockArrangement* ba_;
};

#endif // _MULTIGRID_IMPL_CUDA_H_