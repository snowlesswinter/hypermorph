#ifndef _FLUID_IMPL_CUDA_PURE_H_
#define _FLUID_IMPL_CUDA_PURE_H_

#include <memory>

struct cudaPitchedPtr;
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

    void AdvectVelocity(cudaPitchedPtr* dest, cudaPitchedPtr* velocity,
                        float time_step, float dissipation,
                        const Vectormath::Aos::Vector3& volume_size);

};

#endif // _FLUID_IMPL_CUDA_PURE_H_