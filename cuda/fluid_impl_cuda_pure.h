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
class GraphicsResource;
class FluidImplCudaPure
{
public:
    FluidImplCudaPure();
    ~FluidImplCudaPure();

    void Advect(cudaArray* dest, cudaArray* velocity, cudaArray* source,
                float time_step, float dissipation,
                const Vectormath::Aos::Vector3& volume_size);
    void AdvectDensity(GraphicsResource* dest, cudaArray* velocity,
                       GraphicsResource* density, float time_step,
                       float dissipation,
                       const Vectormath::Aos::Vector3& volume_size);
    void AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                        float time_step, float dissipation,
                        const Vectormath::Aos::Vector3& volume_size);
    void ApplyBuoyancy(cudaArray* dest, cudaArray* velocity,
                       cudaArray* temperature, float time_step,
                       float ambient_temperature, float accel_factor,
                       float gravity,
                       const Vectormath::Aos::Vector3& volume_size);
    void ApplyImpulse(cudaArray* dest, cudaArray* source,
                      const Vectormath::Aos::Vector3& center_point,
                      const Vectormath::Aos::Vector3& hotspot, float radius,
                      float value, const Vectormath::Aos::Vector3& volume_size);
    void ApplyImpulseDensity(GraphicsResource* dest, GraphicsResource* density,
                             const Vectormath::Aos::Vector3& center_point,
                             const Vectormath::Aos::Vector3& hotspot,
                             float radius, float value,
                             const Vectormath::Aos::Vector3& volume_size);
    void ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                           float half_inverse_cell_size,
                           const Vectormath::Aos::Vector3& volume_size);
    void DampedJacobi(cudaArray* dest, cudaArray* packed,
                      float one_minus_omega, float minus_square_cell_size,
                      float omega_over_beta,
                      const Vectormath::Aos::Vector3& volume_size);
    void SubstractGradient(cudaArray* dest, cudaArray* packed,
                           float gradient_scale,
                           const Vectormath::Aos::Vector3& volume_size);
};

#endif // _FLUID_IMPL_CUDA_PURE_H_