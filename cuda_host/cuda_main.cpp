#include "stdafx.h"
#include "cuda_main.h"

#include <cassert>
#include <algorithm>

#include "cuda/cuda_core.h"
#include "cuda/fluid_impl_cuda.h"
#include "cuda/fluid_impl_cuda_pure.h"
#include "cuda/graphics_resource.h"
#include "cuda/multigrid_impl_cuda.h"
#include "cuda_volume.h"
#include "opengl/gl_texture.h"
#include "vmath.hpp"
#include "utility.h"

// =============================================================================
std::pair<GLuint, GraphicsResource*> GetPBO(CudaCore* core, int n,
                                            int num_component,
                                            int width_in_bytes)
{
    static std::pair<GLuint, GraphicsResource*> pixel_buffer[10][5][5] = {};
    std::pair<GLuint, GraphicsResource*>& ref =
        pixel_buffer[n][num_component][width_in_bytes];
    if (!ref.first)
    {
        int width = 128 / n;
        size_t size = width * width * width * num_component * width_in_bytes;

        // create buffer object
        glGenBuffers(1, &(ref.first));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ref.first);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Believe or not: using another CudaCore instance would cause
        //                 cudaGraphicsMapResources() crash or returning
        //                 unknown error!
        //                 This shit just tortured me for a whole day.
        //
        // So, don't treat .cu file as normal cpp files, CUDA must has done
        // something dirty with it. Just put as less as cpp code inside it
        // as possible.

        ref.second = new GraphicsResource(core);
        core->RegisterGLBuffer(ref.first, ref.second);
    }

    return ref;
}
// =============================================================================

namespace
{
void FlushPBO(GLuint pbo, GLuint format, GLTexture* dest, bool half)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    glBindTexture(GL_TEXTURE_3D, dest->handle());
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0,
                    dest->width(), dest->height(), dest->depth(), format,
                    half ? GL_HALF_FLOAT : GL_FLOAT, nullptr);
    assert(glGetError() == 0);
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

vmath::Vector3 FromIntValues(int x, int y, int z)
{
    return vmath::Vector3(static_cast<float>(x), static_cast<float>(y),
                          static_cast<float>(z));
}
} // Anonymous namespace

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
    delete Instance();
}

CudaMain::CudaMain()
    : core_(new CudaCore())
    , fluid_impl_(new FluidImplCuda())
    , fluid_impl_pure_(new FluidImplCudaPure())
    , multigrid_impl_pure_(new MultigridImplCuda())
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
    int r = core_->RegisterGLImage(texture->handle(), texture->target(),
                                   g.get());
    if (r)
        return r;

    registerd_textures_.insert(std::make_pair(texture, std::move(g)));
    return 0;
}

void CudaMain::ProlongatePacked(std::shared_ptr<GLTexture> coarse,
                                std::shared_ptr<GLTexture> fine)
{
    auto i = registerd_textures_.find(coarse);
    auto j = registerd_textures_.find(fine);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    int n = 128 / fine->width();
    auto pbo = GetPBO(core_.get(), n, 4, 4);
    vmath::Vector3 v = FromIntValues(fine->width(), fine->height(),
                                     fine->depth());
    multigrid_impl_pure_->ProlongatePacked(i->second.get(), j->second.get(),
                                           pbo.second, v);

    FlushPBO(pbo.first, GL_RGBA, fine.get(), false);
}

void CudaMain::AdvectVelocity(std::shared_ptr<GLTexture> velocity,
                              std::shared_ptr<GLTexture> dest, float time_step,
                              float dissipation)
{
    auto i = registerd_textures_.find(velocity);
    if (i == registerd_textures_.end())
        return;

    int n = 128 / velocity->width();
    auto pbo = GetPBO(core_.get(), n, 4, 2);
    vmath::Vector3 v = FromIntValues(velocity->width(), velocity->height(),
                                     velocity->depth());
    fluid_impl_->AdvectVelocity(i->second.get(), pbo.second, time_step,
                                dissipation, v);

    FlushPBO(pbo.first, GL_RGBA, dest.get(), true);
}

void CudaMain::Advect(std::shared_ptr<GLTexture> velocity,
                      std::shared_ptr<GLTexture> source,
                      std::shared_ptr<GLTexture> dest, float time_step,
                      float dissipation)
{
    auto i = registerd_textures_.find(velocity);
    auto j = registerd_textures_.find(source);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    int n = 128 / velocity->width();
    auto pbo = GetPBO(core_.get(), n, 1, 2);
    vmath::Vector3 v = FromIntValues(velocity->width(), velocity->height(),
                                     velocity->depth());
    fluid_impl_->Advect(i->second.get(), j->second.get(), pbo.second, time_step,
                        dissipation, v);

    FlushPBO(pbo.first, GL_RED, dest.get(), true);
}

void CudaMain::RoundPassed(int round)
{
    fluid_impl_->RoundPassed(round);
}

void CudaMain::Sync()
{
    core_->Sync();
}

void CudaMain::ApplyBuoyancy(std::shared_ptr<GLTexture> velocity,
                             std::shared_ptr<GLTexture> temperature,
                             std::shared_ptr<GLTexture> dest, float time_step,
                             float ambient_temperature, float accel_factor,
                             float gravity)
{
    auto i = registerd_textures_.find(velocity);
    auto j = registerd_textures_.find(temperature);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    int n = 128 / velocity->width();
    auto pbo = GetPBO(core_.get(), n, 4, 2);
    vmath::Vector3 v = FromIntValues(velocity->width(), velocity->height(),
                                     velocity->depth());
    fluid_impl_->ApplyBuoyancy(i->second.get(), j->second.get(), pbo.second,
                               time_step, ambient_temperature, accel_factor,
                               gravity, v);

    FlushPBO(pbo.first, GL_RGBA, dest.get(), true);
}

void CudaMain::ApplyImpulse(std::shared_ptr<GLTexture> dest,
                            const vmath::Vector3& center_point,
                            const vmath::Vector3& hotspot, float radius,
                            float value)
{
    auto i = registerd_textures_.find(dest);
    if (i == registerd_textures_.end())
        return;

    int n = 128 / dest->width();
    auto pbo = GetPBO(core_.get(), n, 1, 2);
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_->ApplyImpulse(i->second.get(), pbo.second, center_point,
                              hotspot, radius, value, v);

    FlushPBO(pbo.first, GL_RED, dest.get(), true);
}

void CudaMain::ComputeDivergence(std::shared_ptr<GLTexture> velocity,
                                 std::shared_ptr<GLTexture> dest,
                                 float half_inverse_cell_size)
{
    auto i = registerd_textures_.find(velocity);
    if (i == registerd_textures_.end())
        return;

    int n = 128 / dest->width();
    auto pbo = GetPBO(core_.get(), n, 4, 2);
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_->ComputeDivergence(i->second.get(), pbo.second,
                                   half_inverse_cell_size, v);

    FlushPBO(pbo.first, GL_RGBA, dest.get(), true);
}

void CudaMain::DampedJacobi(std::shared_ptr<GLTexture> packed,
                            std::shared_ptr<GLTexture> dest,
                            float one_minus_omega, float minus_square_cell_size,
                            float omega_over_beta)
{
    auto i = registerd_textures_.find(packed);
    if (i == registerd_textures_.end())
        return;

    int n = 128 / dest->width();
    auto pbo = GetPBO(core_.get(), n, 4, 2);
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_->DampedJacobi(i->second.get(), pbo.second, one_minus_omega,
                              minus_square_cell_size, omega_over_beta, v);

    FlushPBO(pbo.first, GL_RGBA, dest.get(), true);
}

void CudaMain::SubstractGradient(std::shared_ptr<GLTexture> velocity,
                                 std::shared_ptr<GLTexture> packed,
                                 std::shared_ptr<GLTexture> dest,
                                 float gradient_scale)
{
    auto i = registerd_textures_.find(velocity);
    auto j = registerd_textures_.find(packed);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    int n = 128 / dest->width();
    auto pbo = GetPBO(core_.get(), n, 4, 2);
    vmath::Vector3 v = FromIntValues(velocity->width(), velocity->height(),
                                     velocity->depth());
    fluid_impl_->SubstractGradient(i->second.get(), j->second.get(), pbo.second,
                                   gradient_scale, v);

    FlushPBO(pbo.first, GL_RGBA, dest.get(), true);
}

void CudaMain::AdvectDensityPure(std::shared_ptr<GLTexture> dest,
                                 std::shared_ptr<CudaVolume> velocity,
                                 std::shared_ptr<GLTexture> density,
                                 float time_step, float dissipation)
{
    auto i = registerd_textures_.find(dest);
    auto j = registerd_textures_.find(density);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->AdvectDensity(i->second.get(), velocity->dev_array(),
                                    j->second.get(), time_step, dissipation, v);
}

void CudaMain::AdvectPure(std::shared_ptr<CudaVolume> dest,
                          std::shared_ptr<CudaVolume> velocity,
                          std::shared_ptr<CudaVolume> source, float time_step,
                          float dissipation)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->Advect(dest->dev_array(), velocity->dev_array(),
                             source->dev_array(), time_step, dissipation, v);
}

void CudaMain::AdvectVelocityPure(std::shared_ptr<CudaVolume> dest,
                                  std::shared_ptr<CudaVolume> velocity,
                                  float time_step, float dissipation)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->AdvectVelocity(dest->dev_array(), velocity->dev_array(),
                                     time_step, dissipation, v);
}

void CudaMain::ApplyBuoyancyPure(std::shared_ptr<CudaVolume> dest,
                                 std::shared_ptr<CudaVolume> velocity,
                                 std::shared_ptr<CudaVolume> temperature,
                                 float time_step, float ambient_temperature,
                                 float accel_factor, float gravity)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->ApplyBuoyancy(dest->dev_array(), velocity->dev_array(),
                                    temperature->dev_array(), time_step,
                                    ambient_temperature, accel_factor, gravity,
                                    v);
}

void CudaMain::ApplyImpulseDensityPure(std::shared_ptr<GLTexture> dest,
                                       std::shared_ptr<GLTexture> density,
                                       const vmath::Vector3& center_point,
                                       const vmath::Vector3& hotspot,
                                       float radius, float value)
{
    auto i = registerd_textures_.find(dest);
    auto j = registerd_textures_.find(density);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->ApplyImpulseDensity(i->second.get(), j->second.get(),
                                          center_point, hotspot, radius, value,
                                          v);
}

void CudaMain::ApplyImpulsePure(std::shared_ptr<CudaVolume> dest,
                                std::shared_ptr<CudaVolume> source,
                                const vmath::Vector3& center_point,
                                const vmath::Vector3& hotspot,
                                float radius, float value)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->ApplyImpulse(dest->dev_array(), source->dev_array(),
                                   center_point, hotspot, radius, value, v);
}

void CudaMain::ComputeDivergencePure(std::shared_ptr<CudaVolume> dest,
                                     std::shared_ptr<CudaVolume> velocity,
                                     float half_inverse_cell_size)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->ComputeDivergence(dest->dev_array(),
                                        velocity->dev_array(),
                                        half_inverse_cell_size, v);
}

void CudaMain::ComputeResidualPackedDiagnosis(
    std::shared_ptr<CudaVolume> dest, std::shared_ptr<CudaVolume> source,
    float inverse_h_square)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
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

void CudaMain::DampedJacobiPure(std::shared_ptr<CudaVolume> packed,
                                float one_minus_omega,
                                float minus_square_cell_size,
                                float omega_over_beta)
{
    vmath::Vector3 v = FromIntValues(packed->width(), packed->height(),
                                     packed->depth());
    fluid_impl_pure_->DampedJacobi(packed->dev_array(), one_minus_omega,
                                   minus_square_cell_size, omega_over_beta, v);
}

void CudaMain::SubstractGradientPure(std::shared_ptr<CudaVolume> dest,
                                     std::shared_ptr<CudaVolume> packed,
                                     float gradient_scale)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    fluid_impl_pure_->SubstractGradient(dest->dev_array(), packed->dev_array(),
                                        gradient_scale, v);
}

void CudaMain::ComputeResidualPackedPure(std::shared_ptr<CudaVolume> dest,
                                         std::shared_ptr<CudaVolume> packed,
                                         float inverse_h_square)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    multigrid_impl_pure_->ComputeResidualPackedPure(dest->dev_array(),
                                                    packed->dev_array(),
                                                    inverse_h_square, v);
}

void CudaMain::ProlongatePackedPure(std::shared_ptr<CudaVolume> coarse,
                                    std::shared_ptr<CudaVolume> fine)
{
    vmath::Vector3 v = FromIntValues(fine->width(), fine->height(),
                                     fine->depth());
    multigrid_impl_pure_->ProlongatePackedPure(fine->dev_array(),
                                               coarse->dev_array(),
                                               fine->dev_array(), v);
}

void CudaMain::RelaxWithZeroGuessPackedPure(std::shared_ptr<CudaVolume> dest,
                                            std::shared_ptr<CudaVolume> packed,
                                            float alpha_omega_over_beta,
                                            float one_minus_omega,
                                            float minus_h_square,
                                            float omega_times_inverse_beta)
{
    vmath::Vector3 v = FromIntValues(dest->width(), dest->height(),
                                     dest->depth());
    multigrid_impl_pure_->RelaxWithZeroGuessPackedPure(dest->dev_array(),
                                                       packed->dev_array(),
                                                       alpha_omega_over_beta,
                                                       one_minus_omega,
                                                       minus_h_square,
                                                       omega_times_inverse_beta,
                                                       v);
}

void CudaMain::RestrictPackedPure(std::shared_ptr<CudaVolume> coarse,
                                  std::shared_ptr<CudaVolume> fine)
{
    vmath::Vector3 v = FromIntValues(coarse->width(), coarse->height(),
                                     coarse->depth());
    multigrid_impl_pure_->RestrictPackedPure(coarse->dev_array(),
                                             fine->dev_array(), v);
}

void CudaMain::RestrictResidualPackedPure(std::shared_ptr<CudaVolume> coarse,
                                          std::shared_ptr<CudaVolume> fine)
{
    vmath::Vector3 v = FromIntValues(coarse->width(), coarse->height(),
                                     coarse->depth());
    multigrid_impl_pure_->RestrictResidualPackedPure(coarse->dev_array(),
                                                     fine->dev_array(), v);
}
