#ifndef _FLUID_IMPL_CUDA_PURE_H_
#define _FLUID_IMPL_CUDA_PURE_H_

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
class FluidImplCudaPure
{
public:
    FluidImplCudaPure();
    ~FluidImplCudaPure();

    void AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                        float time_step, float dissipation,
                        const Vectormath::Aos::Vector3& volume_size);
    void Advect(cudaArray* dest, cudaArray* velocity, cudaArray* source,
                float time_step, float dissipation,
                const Vectormath::Aos::Vector3& volume_size);

};

#endif // _FLUID_IMPL_CUDA_PURE_H_