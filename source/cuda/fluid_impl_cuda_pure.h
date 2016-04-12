#ifndef _FLUID_IMPL_CUDA_PURE_H_
#define _FLUID_IMPL_CUDA_PURE_H_

#include <array>
#include <memory>

struct cudaArray;
namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class BlockArrangement;
class CudaVolume;
class GraphicsResource;
class FluidImplCudaPure
{
public:
    explicit FluidImplCudaPure(BlockArrangement* ba);
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
                      const std::array<float, 3>& value, uint32_t mask,
                      const Vectormath::Aos::Vector3& volume_size);
    void ApplyImpulseDensity(GraphicsResource* density,
                             const Vectormath::Aos::Vector3& center_point,
                             const Vectormath::Aos::Vector3& hotspot,
                             float radius, float value,
                             const Vectormath::Aos::Vector3& volume_size);
    void ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                           float half_inverse_cell_size,
                           const Vectormath::Aos::Vector3& volume_size);
    void ComputeResidualPackedDiagnosis(
        cudaArray* dest, cudaArray* source, float inverse_h_square,
        const Vectormath::Aos::Vector3& volume_size);
    void DampedJacobi(cudaArray* dest, cudaArray* source,
                      float minus_square_cell_size, float omega_over_beta,
                      const Vectormath::Aos::Vector3& volume_size);
    void SubtractGradient(cudaArray* dest, cudaArray* packed,
                          float gradient_scale,
                          const Vectormath::Aos::Vector3& volume_size);

    // For debugging.
    void RoundPassed(int round);

private:
    BlockArrangement* ba_;
};

#endif // _FLUID_IMPL_CUDA_PURE_H_