#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

#include <map>
#include <memory>

#include "third_party/glm/fwd.hpp"

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
    void Advect(std::shared_ptr<CudaVolume> dest,
                std::shared_ptr<CudaVolume> velocity,
                std::shared_ptr<CudaVolume> source, float time_step,
                float dissipation);
    void AdvectVelocity(std::shared_ptr<CudaVolume> dest,
                        std::shared_ptr<CudaVolume> velocity,
                        std::shared_ptr<CudaVolume> velocity_prev,
                        float time_step, float time_step_prev,
                        float dissipation);
    void ApplyBuoyancy(std::shared_ptr<CudaVolume> dest,
                       std::shared_ptr<CudaVolume> velocity,
                       std::shared_ptr<CudaVolume> temperature,
                       float time_step, float ambient_temperature,
                       float accel_factor, float gravity);
    void ApplyImpulseDensity(std::shared_ptr<CudaVolume> density,
                             const glm::vec3& center_point,
                             const glm::vec3& hotspot, float radius,
                             float value);
    void ApplyImpulse(std::shared_ptr<CudaVolume> dest,
                      std::shared_ptr<CudaVolume> source,
                      const glm::vec3& center_point,
                      const glm::vec3& hotspot, float radius,
                      const glm::vec3& value, uint32_t mask);
    void ComputeDivergence(std::shared_ptr<CudaVolume> dest,
                           std::shared_ptr<CudaVolume> velocity,
                           float half_inverse_cell_size);
    void ComputeResidualPackedDiagnosis(std::shared_ptr<CudaVolume> dest,
                                        std::shared_ptr<CudaVolume> velocity,
                                        float inverse_h_square);
    void DampedJacobi(std::shared_ptr<CudaVolume> dest,
                      std::shared_ptr<CudaVolume> source,
                      float minus_square_cell_size, float omega_over_beta,
                      int num_of_iterations);
    void ReviseDensity(std::shared_ptr<CudaVolume> density,
                       const glm::vec3& center_point, float radius,
                       float value);
    void SubtractGradient(std::shared_ptr<CudaVolume> dest,
                          std::shared_ptr<CudaVolume> packed,
                          float gradient_scale);

    // Multigrid.
    void ComputeResidualPacked(std::shared_ptr<CudaVolume> dest,
                               std::shared_ptr<CudaVolume> packed,
                               float inverse_h_square);
    void ProlongatePacked(std::shared_ptr<CudaVolume> coarse,
                          std::shared_ptr<CudaVolume> fine, float overlay);
    void RelaxWithZeroGuessPacked(std::shared_ptr<CudaVolume> dest,
                                  std::shared_ptr<CudaVolume> packed,
                                  float alpha_omega_over_beta,
                                  float one_minus_omega,
                                  float minus_h_square,
                                  float omega_times_inverse_beta);
    void RestrictPacked(std::shared_ptr<CudaVolume> coarse,
                        std::shared_ptr<CudaVolume> fine);
    void RestrictResidualPacked(std::shared_ptr<CudaVolume> coarse,
                                std::shared_ptr<CudaVolume> fine);

    // Rendering
    void Raycast(std::shared_ptr<GLSurface> dest,
                 std::shared_ptr<CudaVolume> density,
                 const glm::mat4& model_view, const glm::vec3& eye_pos,
                 const glm::vec3& light_color, float light_intensity,
                 float focal_length, int num_samples, int num_light_samples,
                 float absorption, float density_factor,
                 float occlusion_factor);

    // For diagnosis
    void RoundPassed(int round);
    void Sync();

private:
    std::unique_ptr<CudaCore> core_;
    std::unique_ptr<FluidImplCuda> fluid_impl_;
    std::unique_ptr<MultigridImplCuda> multigrid_impl_;
    std::map<std::shared_ptr<GLTexture>, std::unique_ptr<GraphicsResource>>
        registerd_textures_;
};

#endif // _CUDA_MAIN_H_