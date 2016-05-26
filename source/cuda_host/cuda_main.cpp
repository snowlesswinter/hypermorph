#include "stdafx.h"
#include "cuda_main.h"

#include <cassert>
#include <algorithm>

#include "cuda/cuda_core.h"
#include "cuda/fluid_impl_cuda.h"
#include "cuda/graphics_resource.h"
#include "cuda/multigrid_impl_cuda.h"
#include "cuda_volume.h"
#include "opengl/gl_surface.h"
#include "opengl/gl_volume.h"
#include "utility.h"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

namespace
{
::AdvectionMethod ToCudaAdvectionMethod(CudaMain::AdvectionMethod method)
{
    switch (method) {
        case CudaMain::SEMI_LAGRANGIAN:
            return ::SEMI_LAGRANGIAN;
        case CudaMain::MACCORMACK_SEMI_LAGRANGIAN:
            return ::MACCORMACK_SEMI_LAGRANGIAN;
        case CudaMain::BFECC_SEMI_LAGRANGIAN:
            return ::BFECC_SEMI_LAGRANGIAN;
        default:
            break;
    }

    return ::INVALID_ADVECTION_METHOD;
}
} // Anonymous namespace.

extern int size_tweak;
CudaMain* CudaMain::Instance()
{
    static CudaMain* instance = nullptr;
    if (!instance) {
        instance = new CudaMain();
        instance->Init();
    }

    return instance;
}

void CudaMain::DestroyInstance()
{
    Instance()->core_->FlushProfilingData();
    delete Instance();
}

CudaMain::CudaMain()
    : core_(new CudaCore())
    , fluid_impl_(new FluidImplCuda(core_->block_arrangement()))
    , multigrid_impl_(new MultigridImplCuda(core_->block_arrangement()))
    , registerd_textures_()
{

}

CudaMain::~CudaMain()
{
}

bool CudaMain::Init()
{
    return core_->Init();
}

int CudaMain::RegisterGLImage(std::shared_ptr<GLTexture> texture)
{
    if (registerd_textures_.find(texture) != registerd_textures_.end())
        return 0;

    std::unique_ptr<GraphicsResource> g(new GraphicsResource(core_.get()));
    int r = core_->RegisterGLImage(texture->texture_handle(), texture->target(),
                                   g.get());
    if (r)
        return r;

    registerd_textures_.insert(std::make_pair(texture, std::move(g)));
    return 0;
}

void CudaMain::UnregisterGLImage(std::shared_ptr<GLTexture> texture)
{
    auto i = registerd_textures_.find(texture);
    assert(i != registerd_textures_.end());
    if (i == registerd_textures_.end())
        return;

    core_->UnregisterGLResource(i->second.get());
    registerd_textures_.erase(i);
}

void CudaMain::Advect(std::shared_ptr<CudaVolume> dest,
                      std::shared_ptr<CudaVolume> velocity,
                      std::shared_ptr<CudaVolume> source,
                      std::shared_ptr<CudaVolume> intermediate, float time_step,
                      float dissipation, AdvectionMethod method)
{
    fluid_impl_->Advect(dest->dev_array(), velocity->dev_array(),
                        source->dev_array(), intermediate->dev_array(),
                        time_step, dissipation, dest->size(),
                        ToCudaAdvectionMethod(method));
}

void CudaMain::AdvectDensity(std::shared_ptr<CudaVolume> dest,
                             std::shared_ptr<CudaVolume> velocity,
                             std::shared_ptr<CudaVolume> density,
                             std::shared_ptr<CudaVolume> intermediate,
                             float time_step, float dissipation,
                             AdvectionMethod method)
{
    fluid_impl_->AdvectDensity(dest->dev_array(), velocity->dev_array(),
                               density->dev_array(), intermediate->dev_array(),
                               time_step, dissipation, dest->size(),
                               ToCudaAdvectionMethod(method));
}

void CudaMain::AdvectVelocity(std::shared_ptr<CudaVolume> dest,
                              std::shared_ptr<CudaVolume> velocity,
                              std::shared_ptr<CudaVolume> velocity_prev,
                              float time_step, float time_step_prev,
                              float dissipation, AdvectionMethod method)
{
    fluid_impl_->AdvectVelocity(dest->dev_array(), velocity->dev_array(),
                                velocity_prev->dev_array(), time_step,
                                time_step_prev, dissipation,
                                dest->size() - size_tweak,
                                ToCudaAdvectionMethod(method));
}

void CudaMain::ApplyBuoyancy(std::shared_ptr<CudaVolume> dest,
                             std::shared_ptr<CudaVolume> velocity,
                             std::shared_ptr<CudaVolume> temperature,
                             std::shared_ptr<CudaVolume> density,
                             float time_step, float ambient_temperature,
                             float accel_factor, float gravity)
{
    // NOTE: The temperature's volume size should be used instead of the
    //       velocity's.
    fluid_impl_->ApplyBuoyancy(dest->dev_array(), velocity->dev_array(),
                               temperature->dev_array(), density->dev_array(),
                               time_step, ambient_temperature, accel_factor,
                               gravity, temperature->size());
}

void CudaMain::ApplyImpulseDensity(std::shared_ptr<CudaVolume> density,
                                   const glm::vec3& center_point,
                                   const glm::vec3& hotspot, float radius,
                                   float value)
{
    fluid_impl_->ApplyImpulseDensity(density->dev_array(), center_point,
                                     hotspot, radius, value, density->size());
}

void CudaMain::ApplyImpulse(std::shared_ptr<CudaVolume> dest,
                            std::shared_ptr<CudaVolume> source,
                            const glm::vec3& center_point,
                            const glm::vec3& hotspot, float radius,
                            const glm::vec3& value, uint32_t mask)
{
    fluid_impl_->ApplyImpulse(dest->dev_array(), source->dev_array(),
                              center_point, hotspot, radius, value, mask,
                              dest->size());
}

void CudaMain::ApplyVorticityConfinement(std::shared_ptr<CudaVolume> dest,
                                         std::shared_ptr<CudaVolume> velocity,
                                         std::shared_ptr<CudaVolume> vort_x,
                                         std::shared_ptr<CudaVolume> vort_y,
                                         std::shared_ptr<CudaVolume> vort_z)
{
    fluid_impl_->ApplyVorticityConfinement(dest->dev_array(),
                                           velocity->dev_array(),
                                           vort_x->dev_array(),
                                           vort_y->dev_array(),
                                           vort_z->dev_array(),
                                           dest->size() - size_tweak);
}

void CudaMain::BuildVorticityConfinement(std::shared_ptr<CudaVolume> dest_x,
                                         std::shared_ptr<CudaVolume> dest_y,
                                         std::shared_ptr<CudaVolume> dest_z,
                                         std::shared_ptr<CudaVolume> vort_x,
                                         std::shared_ptr<CudaVolume> vort_y,
                                         std::shared_ptr<CudaVolume> vort_z,
                                         float coeff, float cell_size)
{
    fluid_impl_->BuildVorticityConfinement(dest_x->dev_array(),
                                           dest_y->dev_array(),
                                           dest_z->dev_array(),
                                           vort_x->dev_array(),
                                           vort_y->dev_array(),
                                           vort_z->dev_array(), coeff,
                                           cell_size, dest_x->size());
}

void CudaMain::ComputeCurl(std::shared_ptr<CudaVolume> dest_x,
                           std::shared_ptr<CudaVolume> dest_y,
                           std::shared_ptr<CudaVolume> dest_z,
                           std::shared_ptr<CudaVolume> velocity,
                           float inverse_cell_size)
{
    fluid_impl_->ComputeCurl(dest_x->dev_array(), dest_y->dev_array(),
                             dest_z->dev_array(), velocity->dev_array(),
                             dest_x->dev_array(), dest_y->dev_array(),
                             dest_z->dev_array(), inverse_cell_size,
                             dest_x->size());
}

void CudaMain::ComputeDivergence(std::shared_ptr<CudaVolume> dest,
                                 std::shared_ptr<CudaVolume> velocity,
                                 float half_inverse_cell_size)
{
    fluid_impl_->ComputeDivergence(dest->dev_array(),
                                   velocity->dev_array(),
                                   half_inverse_cell_size, dest->size());
}

void CudaMain::ComputeResidualPackedDiagnosis(
    std::shared_ptr<CudaVolume> dest, std::shared_ptr<CudaVolume> source,
    float inverse_h_square)
{
    fluid_impl_->ComputeResidualPackedDiagnosis(dest->dev_array(),
                                                source->dev_array(),
                                                inverse_h_square, dest->size());

    // =========================================================================
    int w = dest->width();
    int h = dest->height();
    int d = dest->depth();
    int n = 1;
    int element_size = sizeof(float);

    static char* buf = nullptr;
    if (!buf)
        buf = new char[w * h * d * element_size * n];

    memset(buf, 0, w * h * d * element_size * n);
    CudaCore::CopyFromVolume(buf, w * element_size * n, dest->dev_array(),
                             dest->size());

    float* f = (float*)buf;
    double sum = 0.0;
    double q = 0.0;
    double m = 0.0;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                for (int l = 0; l < n; l++) {
                    q = f[i * w * h * n + j * w * n + k * n + l];
                    //if (i == 30 && j == 0 && k == 56)
                    //if (q > 1)
                    sum += q;
                    m = std::max(q, m);
                }
            }
        }
    }

    double avg = sum / (w * h * d);
    PrintDebugString("(CUDA) avg ||r||: %.8f,    max ||r||: %.8f\n", avg, m);
}

void CudaMain::DampedJacobi(std::shared_ptr<CudaVolume> dest,
                            std::shared_ptr<CudaVolume> source,
                            float minus_square_cell_size,
                            float omega_over_beta, int num_of_iterations)
{
    fluid_impl_->DampedJacobi(dest->dev_array(), source->dev_array(),
                              minus_square_cell_size, omega_over_beta,
                              num_of_iterations, dest->size());
}

void CudaMain::ReviseDensity(std::shared_ptr<CudaVolume> density,
                             const glm::vec3& center_point, float radius,
                             float value)
{
    fluid_impl_->ReviseDensity(density->dev_array(), center_point, radius,
                               value, density->size());
}

void CudaMain::SubtractGradient(std::shared_ptr<CudaVolume> dest,
                                std::shared_ptr<CudaVolume> packed,
                                float half_inverse_cell_size)
{
    // NOTE: The pressure's volume size should be used instead of the
    //       velocity's.
    fluid_impl_->SubtractGradient(dest->dev_array(), packed->dev_array(),
                                  half_inverse_cell_size, packed->size());
}

void CudaMain::ComputeResidualPacked(std::shared_ptr<CudaVolume> dest,
                                     std::shared_ptr<CudaVolume> packed,
                                     float inverse_h_square)
{
    multigrid_impl_->ComputeResidualPacked(dest->dev_array(),
                                           packed->dev_array(),
                                           inverse_h_square, dest->size());
}

void CudaMain::ProlongatePacked(std::shared_ptr<CudaVolume> coarse,
                                    std::shared_ptr<CudaVolume> fine,
                                    float overlay)
{
    multigrid_impl_->ProlongatePacked(fine->dev_array(),
                                      coarse->dev_array(),
                                      fine->dev_array(), overlay,
                                      fine->size());
}

void CudaMain::RelaxWithZeroGuessPacked(std::shared_ptr<CudaVolume> dest,
                                        std::shared_ptr<CudaVolume> packed,
                                        float alpha_omega_over_beta,
                                        float one_minus_omega,
                                        float minus_h_square,
                                        float omega_times_inverse_beta)
{
    multigrid_impl_->RelaxWithZeroGuessPacked(dest->dev_array(),
                                              packed->dev_array(),
                                              alpha_omega_over_beta,
                                              one_minus_omega, minus_h_square,
                                              omega_times_inverse_beta,
                                              dest->size());
}

void CudaMain::RestrictPacked(std::shared_ptr<CudaVolume> coarse,
                              std::shared_ptr<CudaVolume> fine)
{
    multigrid_impl_->RestrictPacked(coarse->dev_array(), fine->dev_array(),
                                    coarse->size());
}

void CudaMain::RestrictResidualPacked(std::shared_ptr<CudaVolume> coarse,
                                      std::shared_ptr<CudaVolume> fine)
{
    multigrid_impl_->RestrictResidualPacked(coarse->dev_array(),
                                            fine->dev_array(), coarse->size());
}

void CudaMain::Raycast(std::shared_ptr<GLSurface> dest,
                       std::shared_ptr<CudaVolume> density,
                       const glm::mat4& model_view, const glm::vec3& eye_pos,
                       const glm::vec3& light_color, float light_intensity,
                       float focal_length, int num_samples,
                       int num_light_samples, float absorption,
                       float density_factor, float occlusion_factor)
{
    auto i = registerd_textures_.find(dest);
    assert(i != registerd_textures_.end());
    if (i == registerd_textures_.end())
        return;

    core_->Raycast(i->second.get(), density->dev_array(), model_view,
                   dest->size(), eye_pos, light_color, light_intensity,
                   focal_length, num_samples, num_light_samples, absorption,
                   density_factor, occlusion_factor);
}

void CudaMain::SetStaggered(bool staggered)
{
    fluid_impl_->set_staggered(staggered);
}

void CudaMain::RoundPassed(int round)
{
    fluid_impl_->RoundPassed(round);
}

void CudaMain::Sync()
{
    core_->Sync();
}
