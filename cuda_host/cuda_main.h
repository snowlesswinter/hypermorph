#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

#include <map>
#include <memory>

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class CudaCore;
class CudaVolume;
class FluidImplCuda;
class FluidImplCudaPure;
class GLTexture;
class GraphicsResource;
class CudaMain
{
public:
    static CudaMain* Instance();

    CudaMain();
    ~CudaMain();

    bool Init();
    int RegisterGLImage(std::shared_ptr<GLTexture> texture);
    void Absolute(std::shared_ptr<GLTexture> texture);
    void ProlongatePacked(std::shared_ptr<GLTexture> coarse,
                          std::shared_ptr<GLTexture> fine);
    void AdvectVelocity(std::shared_ptr<GLTexture> velocity,
                        std::shared_ptr<GLTexture> dest, float time_step,
                        float dissipation);
    void Advect(std::shared_ptr<GLTexture> velocity,
                std::shared_ptr<GLTexture> source,
                std::shared_ptr<GLTexture> dest, float time_step,
                float dissipation);
    void ApplyBuoyancy(std::shared_ptr<GLTexture> velocity,
                       std::shared_ptr<GLTexture> temperature,
                       std::shared_ptr<GLTexture> dest, float time_step,
                       float ambient_temperature, float accel_factor,
                       float gravity);
    void ApplyImpulse(std::shared_ptr<GLTexture> dest,
                      const Vectormath::Aos::Vector3& center_point,
                      const Vectormath::Aos::Vector3& hotspot, float radius,
                      float value);
    void ComputeDivergence(std::shared_ptr<GLTexture> velocity,
                           std::shared_ptr<GLTexture> dest,
                           float half_inverse_cell_size);
    void SubstractGradient(std::shared_ptr<GLTexture> velocity,
                           std::shared_ptr<GLTexture> packed,
                           std::shared_ptr<GLTexture> dest,
                           float gradient_scale);
    void DampedJacobi(std::shared_ptr<GLTexture> packed,
                      std::shared_ptr<GLTexture> dest, float one_minus_omega,
                      float minus_square_cell_size, float omega_over_beta);

    // Pure cuda.
    void AdvectVelocityPure(std::shared_ptr<CudaVolume> dest,
                            std::shared_ptr<CudaVolume> velocity,
                            float time_step, float dissipation);

    // For diagnosis
    void RoundPassed(int round);

    // temporary ===============================================================
    CudaCore* core() { return core_.get(); }

private:
    std::unique_ptr<CudaCore> core_;
    std::unique_ptr<FluidImplCuda> fluid_impl_;
    std::unique_ptr<FluidImplCudaPure> fluid_impl_pure_;
    std::map<std::shared_ptr<GLTexture>, std::unique_ptr<GraphicsResource>>
        registerd_textures_;
};

#endif // _CUDA_MAIN_H_