#include "stdafx.h"
#include "cuda_main.h"

#include <cassert>
#include <algorithm>

#include "cuda/cuda_core.h"
#include "cuda/fluid_impl_cuda_pure.h"
#include "cuda/graphics_resource.h"
#include "cuda/multigrid_impl_cuda.h"
#include "cuda_volume.h"
#include "opengl/gl_surface.h"
#include "opengl/gl_volume.h"
#include "utility.h"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

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
    , fluid_impl_pure_(new FluidImplCudaPure(core_->block_arrangement()))
    , multigrid_impl_pure_(new MultigridImplCuda(core_->block_arrangement()))
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

void CudaMain::AdvectDensity(std::shared_ptr<CudaVolume> dest,
                             std::shared_ptr<CudaVolume> velocity,
                             std::shared_ptr<CudaVolume> density,
                             float time_step, float dissipation)
{
    fluid_impl_pure_->AdvectDensity(
        dest->dev_array(), velocity->dev_array(), density->dev_array(),
        time_step, dissipation,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::AdvectPure(std::shared_ptr<CudaVolume> dest,
                          std::shared_ptr<CudaVolume> velocity,
                          std::shared_ptr<CudaVolume> source, float time_step,
                          float dissipation)
{
    fluid_impl_pure_->Advect(
        dest->dev_array(), velocity->dev_array(), source->dev_array(),
        time_step, dissipation,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::AdvectVelocityPure(std::shared_ptr<CudaVolume> dest,
                                  std::shared_ptr<CudaVolume> velocity,
                                  float time_step, float dissipation)
{
    fluid_impl_pure_->AdvectVelocity(
        dest->dev_array(), velocity->dev_array(), time_step, dissipation,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::ApplyBuoyancyPure(std::shared_ptr<CudaVolume> dest,
                                 std::shared_ptr<CudaVolume> velocity,
                                 std::shared_ptr<CudaVolume> temperature,
                                 float time_step, float ambient_temperature,
                                 float accel_factor, float gravity)
{
    fluid_impl_pure_->ApplyBuoyancy(
        dest->dev_array(), velocity->dev_array(), temperature->dev_array(),
        time_step, ambient_temperature, accel_factor, gravity,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::ApplyImpulseDensityPure(std::shared_ptr<CudaVolume> density,
                                       const glm::vec3& center_point,
                                       const glm::vec3& hotspot, float radius,
                                       float value)
{
    glm::ivec3 v(density->width(), density->height(), density->depth());
    fluid_impl_pure_->ApplyImpulseDensity(density->dev_array(), center_point,
                                          hotspot, radius, value, v);
}

void CudaMain::ApplyImpulsePure(std::shared_ptr<CudaVolume> dest,
                                std::shared_ptr<CudaVolume> source,
                                const glm::vec3& center_point,
                                const glm::vec3& hotspot, float radius,
                                const glm::vec3& value, uint32_t mask)
{
    glm::ivec3 v(dest->width(), dest->height(), dest->depth());
    fluid_impl_pure_->ApplyImpulse(dest->dev_array(), source->dev_array(),
                                   center_point, hotspot, radius, value, mask,
                                   v);
}

void CudaMain::ComputeDivergencePure(std::shared_ptr<CudaVolume> dest,
                                     std::shared_ptr<CudaVolume> velocity,
                                     float half_inverse_cell_size)
{
    fluid_impl_pure_->ComputeDivergence(
        dest->dev_array(), velocity->dev_array(), half_inverse_cell_size,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::ComputeResidualPackedDiagnosis(
    std::shared_ptr<CudaVolume> dest, std::shared_ptr<CudaVolume> source,
    float inverse_h_square)
{
    glm::ivec3 v(dest->width(), dest->height(), dest->depth());
    fluid_impl_pure_->ComputeResidualPackedDiagnosis(dest->dev_array(),
                                                     source->dev_array(),
                                                     inverse_h_square, v);

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
    CudaCore::CopyFromVolume(buf, w * element_size * n, dest->dev_array(), v);

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
    fluid_impl_pure_->DampedJacobi(
        dest->dev_array(), source->dev_array(), minus_square_cell_size,
        omega_over_beta, num_of_iterations,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::SubtractGradientPure(std::shared_ptr<CudaVolume> dest,
                                    std::shared_ptr<CudaVolume> packed,
                                    float gradient_scale)
{
    fluid_impl_pure_->SubtractGradient(
        dest->dev_array(), packed->dev_array(), gradient_scale,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::ComputeResidualPackedPure(std::shared_ptr<CudaVolume> dest,
                                         std::shared_ptr<CudaVolume> packed,
                                         float inverse_h_square)
{
    multigrid_impl_pure_->ComputeResidualPackedPure(
        dest->dev_array(), packed->dev_array(), inverse_h_square,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::ProlongatePackedPure(std::shared_ptr<CudaVolume> coarse,
                                    std::shared_ptr<CudaVolume> fine,
                                    float overlay)
{
    multigrid_impl_pure_->ProlongatePackedPure(
        fine->dev_array(), coarse->dev_array(), fine->dev_array(), overlay,
        glm::ivec3(fine->width(), fine->height(), fine->depth()));
}

void CudaMain::RelaxWithZeroGuessPackedPure(std::shared_ptr<CudaVolume> dest,
                                            std::shared_ptr<CudaVolume> packed,
                                            float alpha_omega_over_beta,
                                            float one_minus_omega,
                                            float minus_h_square,
                                            float omega_times_inverse_beta)
{
    multigrid_impl_pure_->RelaxWithZeroGuessPackedPure(
        dest->dev_array(), packed->dev_array(), alpha_omega_over_beta,
        one_minus_omega, minus_h_square, omega_times_inverse_beta,
        glm::ivec3(dest->width(), dest->height(), dest->depth()));
}

void CudaMain::RestrictPackedPure(std::shared_ptr<CudaVolume> coarse,
                                  std::shared_ptr<CudaVolume> fine)
{
    multigrid_impl_pure_->RestrictPackedPure(
        coarse->dev_array(), fine->dev_array(),
        glm::ivec3(coarse->width(), coarse->height(), coarse->depth()));
}

void CudaMain::RestrictResidualPackedPure(std::shared_ptr<CudaVolume> coarse,
                                          std::shared_ptr<CudaVolume> fine)
{
    multigrid_impl_pure_->RestrictResidualPackedPure(
        coarse->dev_array(), fine->dev_array(),
        glm::ivec3(coarse->width(), coarse->height(), coarse->depth()));
}

void CudaMain::Raycast(std::shared_ptr<GLSurface> dest,
                       std::shared_ptr<CudaVolume> density,
                       const glm::mat4& model_view, const glm::vec3& eye_pos,
                       float focal_length)
{
    auto i = registerd_textures_.find(dest);
    assert(i != registerd_textures_.end());
    if (i == registerd_textures_.end())
        return;

    glm::ivec2 surface_size(dest->width(), dest->height());
    core_->Raycast(i->second.get(), density->dev_array(), model_view,
                   surface_size, eye_pos, focal_length);
}

void CudaMain::RoundPassed(int round)
{
    fluid_impl_pure_->RoundPassed(round);
}

void CudaMain::Sync()
{
    core_->Sync();
}
