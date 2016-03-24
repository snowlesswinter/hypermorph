#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}

struct cudaGraphicsResource;
class GraphicsResource;
class CudaCore
{
public:
    CudaCore();
    ~CudaCore();

    bool Init();
    int RegisterGLImage(unsigned int texture, unsigned int target,
                        GraphicsResource* graphics_res);
    int RegisterGLBuffer(unsigned int buffer, GraphicsResource* graphics_res);
    void UnregisterGLImage(GraphicsResource* graphics_res);
    void Absolute(GraphicsResource* graphics_res, unsigned int aa);
    void ProlongatePacked(GraphicsResource* coarse, GraphicsResource* fine,
                          GraphicsResource* out_pbo,
                          const Vectormath::Aos::Vector3& volume_size_fine);
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

    // For diagnosis.
    void RoundPassed(int round);

private:
};

#endif // _CUDA_CORE_H_