#ifndef _MULTIGRID_IMPL_CUDA_H_
#define _MULTIGRID_IMPL_CUDA_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

struct cudaArray;
class AuxBufferManager;
class BlockArrangement;
class CudaVolume;
class GraphicsResource;
class MultigridImplCuda
{
public:
    MultigridImplCuda(BlockArrangement* ba, AuxBufferManager* bm);
    ~MultigridImplCuda();

    // Multigrid.
    void ComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                         float cell_size, const glm::ivec3& volume_size);
    void Prolongate(cudaArray* fine, cudaArray* coarse,
                    const glm::ivec3& volume_size);
    void ProlongateError(cudaArray* fine, cudaArray* coarse,
                         const glm::ivec3& volume_size);
    void RelaxWithZeroGuess(cudaArray* u, cudaArray* b, float cell_size,
                            const glm::ivec3& volume_size);
    void Restrict(cudaArray* coarse, cudaArray* fine,
                  const glm::ivec3& volume_size);

    // Conjugate gradient.
    void ApplyStencil(cudaArray* aux, cudaArray* search, float cell_size,
                      const glm::ivec3& volume_size);
    void ComputeRho(float* rho, cudaArray* aux, cudaArray* r,
                    const glm::ivec3& volume_size);

private:
    BlockArrangement* ba_;
    AuxBufferManager* bm_;
};

#endif // _MULTIGRID_IMPL_CUDA_H_