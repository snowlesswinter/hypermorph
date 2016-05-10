#ifndef _FLUID_IMPL_CUDA_PURE_H_
#define _FLUID_IMPL_CUDA_PURE_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

struct cudaArray;
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
                const glm::ivec3& volume_size);
    void AdvectDensity(cudaArray* dest, cudaArray* velocity, cudaArray* density,
                       float time_step, float dissipation,
                       const glm::ivec3& volume_size);
    void AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                        float time_step, float dissipation,
                        const glm::ivec3& volume_size);
    void ApplyBuoyancy(cudaArray* dest, cudaArray* velocity,
                       cudaArray* temperature, float time_step,
                       float ambient_temperature, float accel_factor,
                       float gravity, const glm::ivec3& volume_size);
    void ApplyImpulse(cudaArray* dest, cudaArray* source,
                      const glm::vec3& center_point,
                      const glm::vec3& hotspot, float radius,
                      const glm::vec3& value, uint32_t mask,
                      const glm::ivec3& volume_size);
    void ApplyImpulseDensity(cudaArray* density, const glm::vec3& center_point,
                             const glm::vec3& hotspot, float radius,
                             float value, const glm::ivec3& volume_size);
    void ComputeDivergence(cudaArray* dest, cudaArray* velocity,
                           float half_inverse_cell_size,
                           const glm::ivec3& volume_size);
    void ComputeResidualPackedDiagnosis(cudaArray* dest, cudaArray* source,
                                        float inverse_h_square,
                                        const glm::ivec3& volume_size);
    void DampedJacobi(cudaArray* dest, cudaArray* source,
                      float minus_square_cell_size, float omega_over_beta,
                      int num_of_iterations,
                      const glm::ivec3& volume_size);
    void SubtractGradient(cudaArray* dest, cudaArray* packed,
                          float gradient_scale,
                          const glm::ivec3& volume_size);

    // For debugging.
    void RoundPassed(int round);

private:
    BlockArrangement* ba_;
};

#endif // _FLUID_IMPL_CUDA_PURE_H_