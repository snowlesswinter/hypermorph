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
    void Prolongate(cudaArray* fine, cudaArray* coarse,
                    const glm::ivec3& volume_size);
    void RelaxWithZeroGuess(cudaArray* u, cudaArray* b, float cell_size,
                            const glm::ivec3& volume_size);
    void Restrict(cudaArray* coarse, cudaArray* fine,
                  const glm::ivec3& volume_size);

private:
    BlockArrangement* ba_;
};

#endif // _MULTIGRID_IMPL_CUDA_H_