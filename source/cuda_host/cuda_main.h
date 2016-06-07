#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

#include <map>
#include <memory>

#include "third_party/glm/fwd.hpp"

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
    enum AdvectionMethod
    {
        SEMI_LAGRANGIAN,
        MACCORMACK_SEMI_LAGRANGIAN,
        BFECC_SEMI_LAGRANGIAN,
    };

    static CudaMain* Instance();
    static void DestroyInstance();

    CudaMain();
    ~CudaMain();

    bool Init();

    void ClearVolume(CudaVolume* dest, const glm::vec4& value,
                     const glm::ivec3& volume_size);
    int RegisterGLImage(std::shared_ptr<GLTexture> texture);
    void UnregisterGLImage(std::shared_ptr<GLTexture> texture);

    void Advect(std::shared_ptr<CudaVolume> dest,
                std::shared_ptr<CudaVolume> velocity,
                std::shared_ptr<CudaVolume> source,
                std::shared_ptr<CudaVolume> intermediate, float time_step,
                float dissipation, AdvectionMethod method);
    void AdvectDensity(std::shared_ptr<CudaVolume> dest,
                       std::shared_ptr<CudaVolume> velocity,
                       std::shared_ptr<CudaVolume> density,
                       std::shared_ptr<CudaVolume> intermediate,
                       float time_step, float dissipation,
                       AdvectionMethod method);
    void AdvectField(std::shared_ptr<CudaVolume> fnp1,
                     std::shared_ptr<CudaVolume> fn,
                     std::shared_ptr<CudaVolume> vel_x,
                     std::shared_ptr<CudaVolume> vel_y,
                     std::shared_ptr<CudaVolume> vel_z,
                     std::shared_ptr<CudaVolume> aux,
                     float time_step, float dissipation);
    void AdvectVelocity(std::shared_ptr<CudaVolume> vnp1_x,
                        std::shared_ptr<CudaVolume> vnp1_y,
                        std::shared_ptr<CudaVolume> vnp1_z,
                        std::shared_ptr<CudaVolume> vn_x,
                        std::shared_ptr<CudaVolume> vn_y,
                        std::shared_ptr<CudaVolume> vn_z,
                        std::shared_ptr<CudaVolume> aux,
                        float time_step, float dissipation,
                        AdvectionMethod method);
    void AdvectVorticity(std::shared_ptr<CudaVolume> vnp1_x,
                         std::shared_ptr<CudaVolume> vnp1_y,
                         std::shared_ptr<CudaVolume> vnp1_z,
                         std::shared_ptr<CudaVolume> vn_x,
                         std::shared_ptr<CudaVolume> vn_y,
                         std::shared_ptr<CudaVolume> vn_z,
                         std::shared_ptr<CudaVolume> vel_x,
                         std::shared_ptr<CudaVolume> vel_y,
                         std::shared_ptr<CudaVolume> vel_z,
                         std::shared_ptr<CudaVolume> aux,
                         float time_step, float dissipation,
                         AdvectionMethod method);
    void ApplyBuoyancy(std::shared_ptr<CudaVolume> vel_x,
                       std::shared_ptr<CudaVolume> vel_y,
                       std::shared_ptr<CudaVolume> vel_z,
                       std::shared_ptr<CudaVolume> temperature,
                       std::shared_ptr<CudaVolume> density, float time_step,
                       float ambient_temperature, float accel_factor,
                       float gravity);
    void ApplyImpulseDensity(std::shared_ptr<CudaVolume> density,
                             const glm::vec3& center_point,
                             const glm::vec3& hotspot, float radius,
                             float value);
    void ApplyImpulse(std::shared_ptr<CudaVolume> dest,
                      std::shared_ptr<CudaVolume> source,
                      const glm::vec3& center_point,
                      const glm::vec3& hotspot, float radius,
                      const glm::vec3& value, uint32_t mask);
    void ApplyVorticityConfinement(std::shared_ptr<CudaVolume> dest,
                                   std::shared_ptr<CudaVolume> velocity,
                                   std::shared_ptr<CudaVolume> vort_x,
                                   std::shared_ptr<CudaVolume> vort_y,
                                   std::shared_ptr<CudaVolume> vort_z);
    void BuildVorticityConfinement(std::shared_ptr<CudaVolume> dest_x,
                                   std::shared_ptr<CudaVolume> dest_y,
                                   std::shared_ptr<CudaVolume> dest_z,
                                   std::shared_ptr<CudaVolume> vort_x,
                                   std::shared_ptr<CudaVolume> vort_y,
                                   std::shared_ptr<CudaVolume> vort_z,
                                   float coeff, float cell_size);
    void ComputeCurl(std::shared_ptr<CudaVolume> dest_x,
                     std::shared_ptr<CudaVolume> dest_y,
                     std::shared_ptr<CudaVolume> dest_z,
                     std::shared_ptr<CudaVolume> velocity,
                     float inverse_cell_size);
    void ComputeDivergence(std::shared_ptr<CudaVolume> div,
                           std::shared_ptr<CudaVolume> vel_x,
                           std::shared_ptr<CudaVolume> vel_y,
                           std::shared_ptr<CudaVolume> vel_z,
                           float half_inverse_cell_size);
    void ComputeResidualDiagnosis(std::shared_ptr<CudaVolume> residual,
                                  std::shared_ptr<CudaVolume> u,
                                  std::shared_ptr<CudaVolume> b,
                                  float inverse_h_square);
    void Relax(std::shared_ptr<CudaVolume> unp1, std::shared_ptr<CudaVolume> un,
               std::shared_ptr<CudaVolume> b, float cell_size,
               int num_of_iterations);
    void ReviseDensity(std::shared_ptr<CudaVolume> density,
                       const glm::vec3& center_point, float radius,
                       float value);
    void SubtractGradient(std::shared_ptr<CudaVolume> vel_x,
                          std::shared_ptr<CudaVolume> vel_y,
                          std::shared_ptr<CudaVolume> vel_z,
                          std::shared_ptr<CudaVolume> pressure,
                          float half_inverse_cell_size);

    // Multigrid.
    void ComputeResidual(std::shared_ptr<CudaVolume> r,
                         std::shared_ptr<CudaVolume> u,
                         std::shared_ptr<CudaVolume> b, float cell_size);
    void Prolongate(std::shared_ptr<CudaVolume> fine,
                    std::shared_ptr<CudaVolume> coarse);
    void RelaxWithZeroGuess(std::shared_ptr<CudaVolume> u,
                            std::shared_ptr<CudaVolume> b,
                            float cell_size);
    void Restrict(std::shared_ptr<CudaVolume> coarse,
                  std::shared_ptr<CudaVolume> fine);

    // Vorticity.
    void AddCurlPsi(std::shared_ptr<CudaVolume> velocity,
                    std::shared_ptr<CudaVolume> psi_x,
                    std::shared_ptr<CudaVolume> psi_y,
                    std::shared_ptr<CudaVolume> psi_z,
                    float cell_size);
    void ComputeDeltaVorticity(std::shared_ptr<CudaVolume> vort_np1_x,
                               std::shared_ptr<CudaVolume> vort_np1_y,
                               std::shared_ptr<CudaVolume> vort_np1_z,
                               std::shared_ptr<CudaVolume> vort_x,
                               std::shared_ptr<CudaVolume> vort_y,
                               std::shared_ptr<CudaVolume> vort_z);
    void ComputeDivergenceForVort(std::shared_ptr<CudaVolume> div,
                                  std::shared_ptr<CudaVolume> velocity,
                                  float cell_size);
    void DecayVortices(std::shared_ptr<CudaVolume> vort_x,
                       std::shared_ptr<CudaVolume> vort_y,
                       std::shared_ptr<CudaVolume> vort_z,
                       std::shared_ptr<CudaVolume> div, float time_step);
    void StretchVortices(std::shared_ptr<CudaVolume> vort_np1_x,
                         std::shared_ptr<CudaVolume> vort_np1_y,
                         std::shared_ptr<CudaVolume> vort_np1_z,
                         std::shared_ptr<CudaVolume> velocity,
                         std::shared_ptr<CudaVolume> vort_x,
                         std::shared_ptr<CudaVolume> vort_y,
                         std::shared_ptr<CudaVolume> vort_z,
                         float cell_size, float time_step);

    // Rendering
    void Raycast(std::shared_ptr<GLSurface> dest,
                 std::shared_ptr<CudaVolume> density,
                 const glm::mat4& model_view, const glm::vec3& eye_pos,
                 const glm::vec3& light_color, float light_intensity,
                 float focal_length, int num_samples, int num_light_samples,
                 float absorption, float density_factor,
                 float occlusion_factor);

    void SetStaggered(bool staggered);

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