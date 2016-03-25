#ifndef _FLUID_IMPL_CUDA_H_
#define _FLUID_IMPL_CUDA_H_

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}

class GraphicsResource;
class FluidImplCuda
{
public:
    FluidImplCuda();
    ~FluidImplCuda();

    void AdvectVelocity(GraphicsResource* velocity, GraphicsResource* out_pbo,
                        float time_step, float dissipation,
                        const Vectormath::Aos::Vector3& volume_size);
    void Advect(GraphicsResource* velocity, GraphicsResource* source,
                GraphicsResource* out_pbo, float time_step, float dissipation,
                const Vectormath::Aos::Vector3& volume_size);
    void ApplyBuoyancy(GraphicsResource* velocity,
                       GraphicsResource* temperature, GraphicsResource* out_pbo,
                       float time_step, float ambient_temperature,
                       float accel_factor, float gravity,
                       const Vectormath::Aos::Vector3& volume_size);
    void ApplyImpulse(GraphicsResource* source, GraphicsResource* out_pbo,
                      const Vectormath::Aos::Vector3& center_point,
                      const Vectormath::Aos::Vector3& hotspot, float radius,
                      float value, const Vectormath::Aos::Vector3& volume_size);
    void ComputeDivergence(GraphicsResource* velocity,
                           GraphicsResource* out_pbo,
                           float half_inverse_cell_size,
                           const Vectormath::Aos::Vector3& volume_size);
    void SubstractGradient(GraphicsResource* velocity, GraphicsResource* packed,
                           GraphicsResource* out_pbo, float gradient_scale,
                           const Vectormath::Aos::Vector3& volume_size);
    void DampedJacobi(GraphicsResource* packed, GraphicsResource* out_pbo,
                      float one_minus_omega, float minus_square_cell_size,
                      float omega_over_beta,
                      const Vectormath::Aos::Vector3& volume_size);

    // For diagnosis.
    void RoundPassed(int round);
};

#endif // _FLUID_IMPL_CUDA_H_