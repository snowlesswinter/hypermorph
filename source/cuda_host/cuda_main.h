//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

#include <map>
#include <memory>

#include "third_party/glm/fwd.hpp"
#include "cuda_host/cuda_linear_mem.h"

class CudaCore;
class CudaMemPiece;
class CudaVolume;
class FlipImplCuda;
class FluidImplCuda;
class GLSurface;
class GLTexture;
class GraphicsResource;
class PoissonImplCuda;
class CudaMain
{
public:
    enum AdvectionMethod
    {
        SEMI_LAGRANGIAN,
        MACCORMACK_SEMI_LAGRANGIAN,
        BFECC_SEMI_LAGRANGIAN,
        FLIP,
    };

    enum FluidImpulse
    {
        IMPULSE_NONE,
        IMPULSE_HOT_FLOOR,
        IMPULSE_SPHERE,
        IMPULSE_BUOYANT_JET,
        IMPULSE_FLYING_BALL,
    };

    struct FlipParticles
    {
        std::shared_ptr<CudaLinearMemU32> particle_index_;
        std::shared_ptr<CudaLinearMemU32> cell_index_;
        std::shared_ptr<CudaLinearMemU32> particle_count_;
        std::shared_ptr<CudaLinearMemU8>  in_cell_index_;
        std::shared_ptr<CudaLinearMemU16> position_x_;
        std::shared_ptr<CudaLinearMemU16> position_y_;
        std::shared_ptr<CudaLinearMemU16> position_z_;
        std::shared_ptr<CudaLinearMemU16> velocity_x_;
        std::shared_ptr<CudaLinearMemU16> velocity_y_;
        std::shared_ptr<CudaLinearMemU16> velocity_z_;
        std::shared_ptr<CudaLinearMemU16> density_;
        std::shared_ptr<CudaLinearMemU16> temperature_;
        std::shared_ptr<CudaMemPiece>     num_of_actives_;
        int                               num_of_particles_;
    };

    static CudaMain* Instance();
    static void DestroyInstance();

    CudaMain();
    ~CudaMain();

    bool Init();

    void ClearVolume(CudaVolume* dest, const glm::vec4& value,
                     const glm::ivec3& volume_size);
    void CopyVolume(std::shared_ptr<CudaVolume> dest,
                    std::shared_ptr<CudaVolume> source);
    int RegisterGLImage(std::shared_ptr<GLTexture> texture);
    void UnregisterGLImage(std::shared_ptr<GLTexture> texture);
    int RegisterGLBuffer(uint32_t vbo);
    void UnregisterGBuffer(uint32_t vbo);

    void AdvectField(std::shared_ptr<CudaVolume> fnp1,
                     std::shared_ptr<CudaVolume> fn,
                     std::shared_ptr<CudaVolume> vel_x,
                     std::shared_ptr<CudaVolume> vel_y,
                     std::shared_ptr<CudaVolume> vel_z,
                     std::shared_ptr<CudaVolume> aux, float time_step,
                     float dissipation);
    void AdvectVelocity(std::shared_ptr<CudaVolume> vnp1_x,
                        std::shared_ptr<CudaVolume> vnp1_y,
                        std::shared_ptr<CudaVolume> vnp1_z,
                        std::shared_ptr<CudaVolume> vn_x,
                        std::shared_ptr<CudaVolume> vn_y,
                        std::shared_ptr<CudaVolume> vn_z,
                        std::shared_ptr<CudaVolume> aux, float time_step,
                        float dissipation);
    void AdvectVorticity(std::shared_ptr<CudaVolume> vnp1_x,
                         std::shared_ptr<CudaVolume> vnp1_y,
                         std::shared_ptr<CudaVolume> vnp1_z,
                         std::shared_ptr<CudaVolume> vn_x,
                         std::shared_ptr<CudaVolume> vn_y,
                         std::shared_ptr<CudaVolume> vn_z,
                         std::shared_ptr<CudaVolume> vel_x,
                         std::shared_ptr<CudaVolume> vel_y,
                         std::shared_ptr<CudaVolume> vel_z,
                         std::shared_ptr<CudaVolume> aux, float time_step,
                         float dissipation);
    void ApplyBuoyancy(std::shared_ptr<CudaVolume> vnp1_x,
                       std::shared_ptr<CudaVolume> vnp1_y,
                       std::shared_ptr<CudaVolume> vnp1_z,
                       std::shared_ptr<CudaVolume> vn_x,
                       std::shared_ptr<CudaVolume> vn_y,
                       std::shared_ptr<CudaVolume> vn_z,
                       std::shared_ptr<CudaVolume> temperature,
                       std::shared_ptr<CudaVolume> density, float time_step,
                       float ambient_temperature, float accel_factor,
                       float gravity);
    void ApplyImpulse(std::shared_ptr<CudaVolume> vnp1_x,
                      std::shared_ptr<CudaVolume> vnp1_y,
                      std::shared_ptr<CudaVolume>vnp1_z,
                      std::shared_ptr<CudaVolume> d_np1,
                      std::shared_ptr<CudaVolume>t_np1,
                      std::shared_ptr<CudaVolume>vel_x,
                      std::shared_ptr<CudaVolume> vel_y,
                      std::shared_ptr<CudaVolume>vel_z,
                      std::shared_ptr<CudaVolume>density,
                      std::shared_ptr<CudaVolume> temperature,
                      const glm::vec3& center_point,
                      const glm::vec3& hotspot, float radius,
                      float vel_value, float d_value, float t_value);
    void ComputeDivergence(std::shared_ptr<CudaVolume> div,
                           std::shared_ptr<CudaVolume> vel_x,
                           std::shared_ptr<CudaVolume> vel_y,
                           std::shared_ptr<CudaVolume> vel_z);
    void Relax(std::shared_ptr<CudaVolume> unp1, std::shared_ptr<CudaVolume> un,
               std::shared_ptr<CudaVolume> b, int num_of_iterations);
    void ReviseDensity(std::shared_ptr<CudaVolume> density,
                       const glm::vec3& center_point, float radius,
                       float value);
    void SubtractGradient(std::shared_ptr<CudaVolume> vel_x,
                          std::shared_ptr<CudaVolume> vel_y,
                          std::shared_ptr<CudaVolume> vel_z,
                          std::shared_ptr<CudaVolume> pressure);

    // Multigrid.
    void ComputeResidual(std::shared_ptr<CudaVolume> r,
                         std::shared_ptr<CudaVolume> u,
                         std::shared_ptr<CudaVolume> b);
    void Prolongate(std::shared_ptr<CudaVolume> fine,
                    std::shared_ptr<CudaVolume> coarse);
    void ProlongateError(std::shared_ptr<CudaVolume> fine,
                         std::shared_ptr<CudaVolume> coarse);
    void RelaxWithZeroGuess(std::shared_ptr<CudaVolume> u,
                            std::shared_ptr<CudaVolume> b);
    void Restrict(std::shared_ptr<CudaVolume> coarse,
                  std::shared_ptr<CudaVolume> fine);

    // Conjugate gradient.
    void ApplyStencil(std::shared_ptr<CudaVolume> aux,
                      std::shared_ptr<CudaVolume> search);
    void ComputeAlpha(std::shared_ptr<CudaMemPiece> alpha,
                      std::shared_ptr<CudaMemPiece> rho,
                      std::shared_ptr<CudaVolume> aux,
                      std::shared_ptr<CudaVolume> search);
    void ComputeRho(std::shared_ptr<CudaMemPiece> rho,
                    std::shared_ptr<CudaVolume> search,
                    std::shared_ptr<CudaVolume> residual);
    void ComputeRhoAndBeta(std::shared_ptr<CudaMemPiece> beta,
                           std::shared_ptr<CudaMemPiece> rho_new,
                           std::shared_ptr<CudaMemPiece> rho,
                           std::shared_ptr<CudaVolume> aux,
                           std::shared_ptr<CudaVolume> residual);
    void ScaledAdd(std::shared_ptr<CudaVolume> dest,
                   std::shared_ptr<CudaVolume> v0,
                   std::shared_ptr<CudaVolume> v1,
                   std::shared_ptr<CudaMemPiece> coef, float sign);
    void ScaleVector(std::shared_ptr<CudaVolume> dest,
                     std::shared_ptr<CudaVolume> v,
                     std::shared_ptr<CudaMemPiece> coef, float sign);

    // Vorticity.
    void AddCurlPsi(std::shared_ptr<CudaVolume> vel_x,
                    std::shared_ptr<CudaVolume> vel_y,
                    std::shared_ptr<CudaVolume> vel_z,
                    std::shared_ptr<CudaVolume> psi_x,
                    std::shared_ptr<CudaVolume> psi_y,
                    std::shared_ptr<CudaVolume> psi_z);
    void ApplyVorticityConfinement(std::shared_ptr<CudaVolume> vel_x,
                                   std::shared_ptr<CudaVolume> vel_y,
                                   std::shared_ptr<CudaVolume> vel_z,
                                   std::shared_ptr<CudaVolume> vort_x,
                                   std::shared_ptr<CudaVolume> vort_y,
                                   std::shared_ptr<CudaVolume> vort_z);
    void BuildVorticityConfinement(std::shared_ptr<CudaVolume> conf_x,
                                   std::shared_ptr<CudaVolume> conf_y,
                                   std::shared_ptr<CudaVolume> conf_z,
                                   std::shared_ptr<CudaVolume> vort_x,
                                   std::shared_ptr<CudaVolume> vort_y,
                                   std::shared_ptr<CudaVolume> vort_z,
                                   float coeff);
    void ComputeCurl(std::shared_ptr<CudaVolume> vort_x,
                     std::shared_ptr<CudaVolume> vort_y,
                     std::shared_ptr<CudaVolume> vort_z,
                     std::shared_ptr<CudaVolume> vel_x,
                     std::shared_ptr<CudaVolume> vel_y,
                     std::shared_ptr<CudaVolume> vel_z);
    void ComputeDeltaVorticity(std::shared_ptr<CudaVolume> delta_x,
                               std::shared_ptr<CudaVolume> delta_y,
                               std::shared_ptr<CudaVolume> delta_z,
                               std::shared_ptr<CudaVolume> vort_x,
                               std::shared_ptr<CudaVolume> vort_y,
                               std::shared_ptr<CudaVolume> vort_z);
    void DecayVortices(std::shared_ptr<CudaVolume> vort_x,
                       std::shared_ptr<CudaVolume> vort_y,
                       std::shared_ptr<CudaVolume> vort_z,
                       std::shared_ptr<CudaVolume> div, float time_step);
    void StretchVortices(std::shared_ptr<CudaVolume> vnp1_x,
                         std::shared_ptr<CudaVolume> vnp1_y,
                         std::shared_ptr<CudaVolume> vnp1_z,
                         std::shared_ptr<CudaVolume> vel_x,
                         std::shared_ptr<CudaVolume> vel_y,
                         std::shared_ptr<CudaVolume> vel_z,
                         std::shared_ptr<CudaVolume> vort_x,
                         std::shared_ptr<CudaVolume> vort_y,
                         std::shared_ptr<CudaVolume> vort_z, float time_step);

    // Particles
    void EmitParticles(FlipParticles* particles, const glm::vec3& center_point,
                       const glm::vec3& hotspot, float radius, float density,
                       float temperature, const glm::vec3& velocity,
                       const glm::ivec3& volume_size);
    void MoveParticles(FlipParticles* particles, int* num_active_particles,
                       const FlipParticles* aux_,
                       std::shared_ptr<CudaVolume> vnp1_x,
                       std::shared_ptr<CudaVolume> vnp1_y,
                       std::shared_ptr<CudaVolume> vnp1_z,
                       std::shared_ptr<CudaVolume> vn_x,
                       std::shared_ptr<CudaVolume> vn_y,
                       std::shared_ptr<CudaVolume> vn_z,
                       std::shared_ptr<CudaVolume> density,
                       std::shared_ptr<CudaVolume> temperature,
                       float velocity_dissipation, float density_dissipation,
                       float temperature_dissipation, float time_step);
    void ResetParticles(FlipParticles* particles,
                        const glm::ivec3& volume_size);

    // Rendering
    bool CopyToVbo(uint32_t point_vbo, uint32_t extra_vbo,
                   std::shared_ptr<CudaLinearMemU16> pos_x,
                   std::shared_ptr<CudaLinearMemU16> pos_y,
                   std::shared_ptr<CudaLinearMemU16> pos_z,
                   std::shared_ptr<CudaLinearMemU16> density,
                   std::shared_ptr<CudaLinearMemU16> temperature,
                   std::shared_ptr<CudaMemPiece> num_of_actives,
                   float crit_density, int num_of_particles);
    void Raycast(std::shared_ptr<GLSurface> dest,
                 std::shared_ptr<CudaVolume> density,
                 const glm::mat4& inv_rotation, const glm::vec3& eye_pos,
                 const glm::vec3& light_color, const glm::vec3& light_pos,
                 float light_intensity, float focal_length,
                 const glm::vec2& screen_size, int num_samples,
                 int num_light_samples, float absorption, float density_factor,
                 float occlusion_factor);

    void SetAdvectionMethod(AdvectionMethod method);
    void SetCellSize(float cell_size);
    void SetFluidImpulse(FluidImpulse impulse);
    void SetMidPoint(bool mid_point);
    void SetOutflow(bool outflow);
    void SetStaggered(bool staggered);

    // For diagnosis
    void ComputeResidualDiagnosis(std::shared_ptr<CudaVolume> residual,
                                  std::shared_ptr<CudaVolume> u,
                                  std::shared_ptr<CudaVolume> b);
    void PrintVolume(std::shared_ptr<CudaVolume> volume,
                     const std::string& name);
    void RoundPassed(int round);
    void Sync();

private:
    class FlipObserver;

    std::unique_ptr<CudaCore> core_;
    std::unique_ptr<FluidImplCuda> fluid_impl_;
    std::unique_ptr<PoissonImplCuda> poisson_impl_;

    std::shared_ptr<FlipObserver> flip_ob_;
    std::unique_ptr<FlipImplCuda> flip_impl_;
    std::map<std::shared_ptr<GLTexture>, std::unique_ptr<GraphicsResource>>
        registerd_textures_;
    std::map<uint32_t, std::unique_ptr<GraphicsResource>> registerd_buffers_;
};

#endif // _CUDA_MAIN_H_