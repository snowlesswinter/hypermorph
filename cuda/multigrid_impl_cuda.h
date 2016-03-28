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

    void Advect(cudaArray* dest, cudaArray* velocity, cudaArray* source,
                float time_step, float dissipation,
                const Vectormath::Aos::Vector3& volume_size);
};

#endif // _MULTIGRID_IMPL_CUDA_H_