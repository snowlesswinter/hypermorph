#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

#include <map>
#include <memory>

#include "third_party/glm/mat4x4.hpp"

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class CudaCore;
class CudaVolume;
class FluidImplCudaPure;
class GLSurface;
class GLTexture;
class GraphicsResource;
class MultigridImplCuda;
class CudaMain
{
public:
    static CudaMain* Instance();
    static void DestroyInstance();

    CudaMain();
    ~CudaMain();

    bool Init();

    int RegisterGLImage(std::shared_ptr<GLTexture> texture);
    void UnregisterGLImage(std::shared_ptr<GLTexture> texture);

    void AdvectDensity(std::shared_ptr<CudaVolume> dest,
                       std::shared_ptr<CudaVolume> velocity,
                       std::shared_ptr<CudaVolume> density, float time_step,
                       float dissipation);
    void AdvectPure(std::shared_ptr<CudaVolume> dest,
                    std::shared_ptr<CudaVolume> velocity,
                    std::shared_ptr<CudaVolume> source, float time_step,
                    float dissipation);
    void AdvectVelocityPure(std::shared_ptr<CudaVolume> dest,
                            std::shared_ptr<CudaVolume> velocity,
                            float time_step, float dissipation);
    void ApplyBuoyancyPure(std::shared_ptr<CudaVolume> dest,
                           std::shared_ptr<CudaVolume> velocity,
                           std::shared_ptr<CudaVolume> temperature,
                           float time_step, float ambient_temperature,
                           float accel_factor, float gravity);
    void ApplyImpulseDensityPure(std::shared_ptr<CudaVolume> density,
                                 const glm::vec3& center_point,
                                 const glm::vec3& hotspot, float radius,
                                 float value);
    void ApplyImpulsePure(std::shared_ptr<CudaVolume> dest,
                          std::shared_ptr<CudaVolume> source,
                          const glm::vec3& center_point,
                          const glm::vec3& hotspot, float radius,
                          const glm::vec3& value, uint32_t mask);
    void ComputeDivergencePure(std::shared_ptr<CudaVolume> dest,
                               std::shared_ptr<CudaVolume> velocity,
                               float half_inverse_cell_size);
    void ComputeResidualPackedDiagnosis(std::shared_ptr<CudaVolume> dest,
                                        std::shared_ptr<CudaVolume> velocity,
                                        float inverse_h_square);
    void DampedJacobi(std::shared_ptr<CudaVolume> dest,
                      std::shared_ptr<CudaVolume> source,
                      float minus_square_cell_size, float omega_over_beta,
                      int num_of_iterations);
    void SubtractGradientPure(std::shared_ptr<CudaVolume> dest,
                              std::shared_ptr<CudaVolume> packed,
                              float gradient_scale);

    // Multigrid.
    void ComputeResidualPackedPure(std::shared_ptr<CudaVolume> dest,
                                   std::shared_ptr<CudaVolume> packed,
                                   float inverse_h_square);
    void ProlongatePackedPure(std::shared_ptr<CudaVolume> coarse,
                              std::shared_ptr<CudaVolume> fine, float overlay);
    void RelaxWithZeroGuessPackedPure(std::shared_ptr<CudaVolume> dest,
                                      std::shared_ptr<CudaVolume> packed,
                                      float alpha_omega_over_beta,
                                      float one_minus_omega,
                                      float minus_h_square,
                                      float omega_times_inverse_beta);
    void RestrictPackedPure(std::shared_ptr<CudaVolume> coarse,
                            std::shared_ptr<CudaVolume> fine);
    void RestrictResidualPackedPure(std::shared_ptr<CudaVolume> coarse,
                                    std::shared_ptr<CudaVolume> fine);

    // Rendering
    void Raycast(std::shared_ptr<GLSurface> dest,
                 std::shared_ptr<CudaVolume> density,
                 const glm::mat4& model_view, const glm::vec3& eye_pos,
                 float focal_length);

    // For diagnosis
    void RoundPassed(int round);
    void Sync();

private:
    std::unique_ptr<CudaCore> core_;
    std::unique_ptr<FluidImplCudaPure> fluid_impl_pure_;
    std::unique_ptr<MultigridImplCuda> multigrid_impl_pure_;
    std::map<std::shared_ptr<GLTexture>, std::unique_ptr<GraphicsResource>>
        registerd_textures_;
};

#endif // _CUDA_MAIN_H_