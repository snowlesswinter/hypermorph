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

    void ComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                         float cell_size, const glm::ivec3& volume_size);
    void ComputeResidualPacked(cudaArray* dest_array, cudaArray* source_array,
                               float inverse_h_square,
                               const glm::ivec3& volume_size);
    void Prolongate(cudaArray* fine, cudaArray* coarse,
                    const glm::ivec3& volume_size);
    void ProlongatePacked(cudaArray* dest, cudaArray* coarse, cudaArray* fine,
                          float overlay, const glm::ivec3& volume_size);
    void RelaxWithZeroGuess(cudaArray* u, cudaArray* b, float cell_size,
                            const glm::ivec3& volume_size);
    void RelaxWithZeroGuessPacked(cudaArray* dest_array,
                                  cudaArray* source_array,
                                  float alpha_omega_over_beta,
                                  float one_minus_omega, float minus_h_square,
                                  float omega_times_inverse_beta,
                                  const glm::ivec3& volume_size);
    void RestrictPacked(cudaArray* dest_array, cudaArray* source_array,
                        const glm::ivec3& volume_size);
    void RestrictResidual(cudaArray* b, cudaArray* r,
                          const glm::ivec3& volume_size);
    void RestrictResidualPacked(cudaArray* dest_array, cudaArray* source_array,
                                const glm::ivec3& volume_size);

private:
    BlockArrangement* ba_;
};

#endif // _MULTIGRID_IMPL_CUDA_H_