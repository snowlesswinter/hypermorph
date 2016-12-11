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

#include "stdafx.h"
#include "cuda_main.h"

#include <cassert>
#include <algorithm>

#include "cuda/cuda_core.h"
#include "cuda/fluid_impl_cuda.h"
#include "cuda/graphics_resource.h"
#include "cuda/mem_piece.h"
#include "cuda/particle/flip.h"
#include "cuda/particle/flip_impl_cuda.h"
#include "cuda/particle/particle_impl_cuda.h"
#include "cuda/poisson_impl_cuda.h"
#include "cuda_mem_piece.h"
#include "cuda_volume.h"
#include "metrics.h" // TODO
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

::FluidImpulse ToCudaFluidImpulse(CudaMain::FluidImpulse impulse)
{
    switch (impulse) {
        case CudaMain::IMPULSE_HOT_FLOOR:
            return ::IMPULSE_HOT_FLOOR;
        case CudaMain::IMPULSE_SPHERE:
            return ::IMPULSE_SPHERE;
        case CudaMain::IMPULSE_BUOYANT_JET:
            return ::IMPULSE_BUOYANT_JET;
        case CudaMain::IMPULSE_FLYING_BALL:
            return ::IMPULSE_FLYING_BALL;
        default:
            break;
    }

    return ::IMPULSE_NONE;
}

::FlipParticles ToCudaFlipParticles(const CudaMain::FlipParticles& p)
{
    ::FlipParticles cuda_p;
    cuda_p.particle_count_   = p.particle_count_ ? p.particle_count_->mem() : nullptr;
    cuda_p.position_x_       = p.position_x_->mem();
    cuda_p.position_y_       = p.position_y_->mem();
    cuda_p.position_z_       = p.position_z_->mem();
    cuda_p.velocity_x_       = p.velocity_x_->mem();
    cuda_p.velocity_y_       = p.velocity_y_->mem();
    cuda_p.velocity_z_       = p.velocity_z_->mem();
    cuda_p.density_          = p.density_->mem();
    cuda_p.temperature_      = p.temperature_->mem();
    cuda_p.num_of_actives_   = p.num_of_actives_ ? reinterpret_cast<int*>(p.num_of_actives_->mem()) : nullptr;
    cuda_p.num_of_particles_ = p.num_of_particles_;
    return cuda_p;
}
} // Anonymous namespace.

class CudaMain::FlipObserver : public FlipImplCuda::Observer
{
public:
    virtual void OnEmitted() override
    {
        Metrics::Instance()->OnParticleEmitted();
    }
    virtual void OnVelocityInterpolated() override
    {
        Metrics::Instance()->OnParticleVelocityInterpolated();
    }
    virtual void OnResampled() override
    {
        Metrics::Instance()->OnParticleResampled();
    }
    virtual void OnAdvected() override
    {
        Metrics::Instance()->OnParticleAdvected();
    }
    virtual void OnCellBound() override
    {
        Metrics::Instance()->OnParticleCellBound();
    }
    virtual void OnPrefixSumCalculated() override
    {
        Metrics::Instance()->OnParticlePrefixSumCalculated();
    }
    virtual void OnSorted() override
    {
        Metrics::Instance()->OnParticleSorted();
    }
    virtual void OnTransferred() override
    {
        Metrics::Instance()->OnParticleTransferred();
    }
};

class CudaMain::ParticleObserver : public ParticleImplCuda::Observer
{
public:
    virtual void OnEmitted() override
    {
        //Metrics::Instance()->OnParticleEmitted();
    }
    virtual void OnAdvected() override
    {
        //Metrics::Instance()->OnParticleAdvected();
    }
};

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
    , poisson_impl_(
        new PoissonImplCuda(core_->block_arrangement(),
                            core_->buffer_manager()))
    , flip_ob_(std::make_shared<FlipObserver>())
    , flip_impl_(
        new FlipImplCuda(flip_ob_.get(), core_->block_arrangement(),
                         core_->buffer_manager(), core_->rand_helper()))
    , particle_ob_(std::make_shared<ParticleObserver>())
    , particle_impl_(
        new ParticleImplCuda(particle_ob_.get(), core_->block_arrangement(),
                             core_->buffer_manager(), core_->rand_helper()))
    , registerd_textures_()
    , registerd_buffers_()
{

}

CudaMain::~CudaMain()
{
}

bool CudaMain::Init()
{
    return core_->Init();
}

void CudaMain::ClearVolume(CudaVolume* dest, const glm::vec4& value,
                           const glm::ivec3& volume_size)
{
    core_->ClearVolume(dest->dev_array(), value, volume_size);
}

void CudaMain::CopyVolume(std::shared_ptr<CudaVolume> dest,
                          std::shared_ptr<CudaVolume> source)
{
    CudaCore::CopyVolumeAsync(dest->dev_array(), source->dev_array(),
                              dest->size());
}

int CudaMain::RegisterGLImage(std::shared_ptr<GLTexture> texture)
{
    if (registerd_textures_.find(texture) != registerd_textures_.end())
        return 0;

    std::unique_ptr<GraphicsResource> g(new GraphicsResource(core_.get()));
    int r = core_->RegisterGLImage(texture->texture_handle(), texture->target(),
                                   g.get());
    assert(!r);
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

    registerd_textures_.erase(i); // The resource will be unregistered in its
                                  // dtor.
}

int CudaMain::RegisterGLBuffer(uint32_t vbo)
{
    if (registerd_buffers_.find(vbo) != registerd_buffers_.end())
        return 0;

    std::unique_ptr<GraphicsResource> g(new GraphicsResource(core_.get()));
    int r = core_->RegisterGLBuffer(vbo, g.get());
    assert(!r);
    if (r)
        return r;

    registerd_buffers_.insert(std::make_pair(vbo, std::move(g)));
    return 0;
}

void CudaMain::UnregisterGBuffer(uint32_t vbo)
{
    auto i = registerd_buffers_.find(vbo);
    assert(i != registerd_buffers_.end());
    if (i == registerd_buffers_.end())
        return;

    registerd_buffers_.erase(i); // The resource will be unregistered in its
                                 // dtor.
}

void CudaMain::AdvectField(std::shared_ptr<CudaVolume> fnp1,
                           std::shared_ptr<CudaVolume> fn,
                           std::shared_ptr<CudaVolume> vel_x,
                           std::shared_ptr<CudaVolume> vel_y,
                           std::shared_ptr<CudaVolume> vel_z,
                           std::shared_ptr<CudaVolume> aux,
                           float time_step, float dissipation)
{
    fluid_impl_->AdvectScalarField(fnp1->dev_array(), fn->dev_array(),
                                   vel_x->dev_array(), vel_y->dev_array(),
                                   vel_z->dev_array(), aux->dev_array(),
                                   time_step, dissipation, fnp1->size());
}

void CudaMain::AdvectVelocity(std::shared_ptr<CudaVolume> vnp1_x,
                              std::shared_ptr<CudaVolume> vnp1_y,
                              std::shared_ptr<CudaVolume> vnp1_z,
                              std::shared_ptr<CudaVolume> vn_x,
                              std::shared_ptr<CudaVolume> vn_y,
                              std::shared_ptr<CudaVolume> vn_z,
                              std::shared_ptr<CudaVolume> aux,
                              float time_step, float dissipation)
{
    fluid_impl_->AdvectVectorFields(vnp1_x->dev_array(), vnp1_y->dev_array(),
                                    vnp1_z->dev_array(), vn_x->dev_array(),
                                    vn_y->dev_array(), vn_z->dev_array(),
                                    vn_x->dev_array(), vn_y->dev_array(),
                                    vn_z->dev_array(), aux->dev_array(),
                                    time_step, dissipation, vnp1_x->size(),
                                    FluidImplCuda::VECTOR_FIELD_VELOCITY);
}

void CudaMain::AdvectVorticity(std::shared_ptr<CudaVolume> vnp1_x,
                               std::shared_ptr<CudaVolume> vnp1_y,
                               std::shared_ptr<CudaVolume> vnp1_z,
                               std::shared_ptr<CudaVolume> vn_x,
                               std::shared_ptr<CudaVolume> vn_y,
                               std::shared_ptr<CudaVolume> vn_z,
                               std::shared_ptr<CudaVolume> vel_x,
                               std::shared_ptr<CudaVolume> vel_y,
                               std::shared_ptr<CudaVolume> vel_z,
                               std::shared_ptr<CudaVolume> aux,
                               float time_step, float dissipation)
{
    fluid_impl_->AdvectVectorFields(vnp1_x->dev_array(), vnp1_y->dev_array(),
                                    vnp1_z->dev_array(), vn_x->dev_array(),
                                    vn_y->dev_array(), vn_z->dev_array(),
                                    vel_x->dev_array(), vel_y->dev_array(),
                                    vel_z->dev_array(), aux->dev_array(), 
                                    time_step, dissipation, vnp1_x->size(),
                                    FluidImplCuda::VECTOR_FIELD_VORTICITY);
}

void CudaMain::ApplyBuoyancy(std::shared_ptr<CudaVolume> vnp1_x,
                             std::shared_ptr<CudaVolume> vnp1_y,
                             std::shared_ptr<CudaVolume> vnp1_z,
                             std::shared_ptr<CudaVolume> vn_x,
                             std::shared_ptr<CudaVolume> vn_y,
                             std::shared_ptr<CudaVolume> vn_z,
                             std::shared_ptr<CudaVolume> temperature,
                             std::shared_ptr<CudaVolume> density,
                             float time_step, float ambient_temperature,
                             float accel_factor, float gravity)
{
    fluid_impl_->ApplyBuoyancy(vnp1_x->dev_array(), vnp1_y->dev_array(),
                               vnp1_z->dev_array(), vn_x->dev_array(),
                               vn_y->dev_array(), vn_z->dev_array(),
                               temperature->dev_array(), density->dev_array(),
                               time_step, ambient_temperature, accel_factor,
                               gravity, vnp1_x->size());
}

void CudaMain::ApplyImpulse(std::shared_ptr<CudaVolume> vnp1_x,
                            std::shared_ptr<CudaVolume> vnp1_y,
                            std::shared_ptr<CudaVolume> vnp1_z,
                            std::shared_ptr<CudaVolume> d_np1,
                            std::shared_ptr<CudaVolume> t_np1,
                            std::shared_ptr<CudaVolume> vel_x,
                            std::shared_ptr<CudaVolume> vel_y,
                            std::shared_ptr<CudaVolume> vel_z,
                            std::shared_ptr<CudaVolume> density,
                            std::shared_ptr<CudaVolume> temperature,
                            const glm::vec3& center_point,
                            const glm::vec3& hotspot, float radius,
                            const glm::vec3& vel_value, float d_value,
                            float t_value)
{
    fluid_impl_->ApplyImpulse(vnp1_x->dev_array(), vnp1_y->dev_array(),
                              vnp1_z->dev_array(), d_np1->dev_array(),
                              t_np1->dev_array(), vel_x->dev_array(),
                              vel_y->dev_array(), vel_z->dev_array(),
                              density->dev_array(), temperature->dev_array(),
                              center_point, hotspot, radius, vel_value,
                              d_value, t_value, vnp1_x->size());
}

void CudaMain::ComputeDivergence(std::shared_ptr<CudaVolume> div,
                                 std::shared_ptr<CudaVolume> vel_x,
                                 std::shared_ptr<CudaVolume> vel_y,
                                 std::shared_ptr<CudaVolume> vel_z)
{
    fluid_impl_->ComputeDivergence(div->dev_array(), vel_x->dev_array(),
                                   vel_y->dev_array(), vel_z->dev_array(),
                                   div->size());
}

void CudaMain::Relax(std::shared_ptr<CudaVolume> unp1,
                     std::shared_ptr<CudaVolume> un,
                     std::shared_ptr<CudaVolume> b, int num_of_iterations)
{
    fluid_impl_->Relax(unp1->dev_array(), un->dev_array(), b->dev_array(),
                       num_of_iterations, unp1->size());
}

void CudaMain::ReviseDensity(std::shared_ptr<CudaVolume> density,
                             const glm::vec3& center_point, float radius,
                             float value)
{
    fluid_impl_->ReviseDensity(density->dev_array(), center_point, radius,
                               value, density->size());
}

void CudaMain::SubtractGradient(std::shared_ptr<CudaVolume> vel_x,
                                std::shared_ptr<CudaVolume> vel_y,
                                std::shared_ptr<CudaVolume> vel_z,
                                std::shared_ptr<CudaVolume> pressure)
{
    fluid_impl_->SubtractGradient(vel_x->dev_array(), vel_y->dev_array(),
                                  vel_z->dev_array(), pressure->dev_array(),
                                  vel_x->size());
}

void CudaMain::ComputeResidual(std::shared_ptr<CudaVolume> r,
                               std::shared_ptr<CudaVolume> u,
                               std::shared_ptr<CudaVolume> b)
{
    poisson_impl_->ComputeResidual(r->dev_array(), u->dev_array(),
                                   b->dev_array(), r->size());
}

void CudaMain::Prolongate(std::shared_ptr<CudaVolume> fine,
                          std::shared_ptr<CudaVolume> coarse)
{
    poisson_impl_->Prolongate(fine->dev_array(), coarse->dev_array(),
                              fine->size());
}

void CudaMain::ProlongateError(std::shared_ptr<CudaVolume> fine,
                               std::shared_ptr<CudaVolume> coarse)
{
    poisson_impl_->ProlongateError(fine->dev_array(), coarse->dev_array(),
                                   fine->size());
}

void CudaMain::RelaxWithZeroGuess(std::shared_ptr<CudaVolume> u,
                                  std::shared_ptr<CudaVolume> b)
{
    poisson_impl_->RelaxWithZeroGuess(u->dev_array(), b->dev_array(),
                                      u->size());
}

void CudaMain::Restrict(std::shared_ptr<CudaVolume> coarse,
                        std::shared_ptr<CudaVolume> fine)
{
    poisson_impl_->Restrict(coarse->dev_array(), fine->dev_array(),
                            coarse->size());
}

void CudaMain::ApplyStencil(std::shared_ptr<CudaVolume> aux,
                            std::shared_ptr<CudaVolume> search)
{
    poisson_impl_->ApplyStencil(aux->dev_array(), search->dev_array(),
                                aux->size());
}

void CudaMain::ComputeAlpha(std::shared_ptr<CudaMemPiece> alpha,
                            std::shared_ptr<CudaMemPiece> rho,
                            std::shared_ptr<CudaVolume> aux,
                            std::shared_ptr<CudaVolume> search)
{
    poisson_impl_->ComputeAlpha(MemPiece(alpha->mem(), alpha->size()),
                                MemPiece(rho->mem(), rho->size()),
                                aux->dev_array(), search->dev_array(),
                                aux->size());
}

void CudaMain::ComputeRho(std::shared_ptr<CudaMemPiece> rho,
                          std::shared_ptr<CudaVolume> search,
                          std::shared_ptr<CudaVolume> residual)
{
    poisson_impl_->ComputeRho(
        MemPiece(rho->mem(), rho->size()), search->dev_array(),
        residual->dev_array(), search->size());
}

void CudaMain::ComputeRhoAndBeta(std::shared_ptr<CudaMemPiece> beta,
                                 std::shared_ptr<CudaMemPiece> rho_new,
                                 std::shared_ptr<CudaMemPiece> rho,
                                 std::shared_ptr<CudaVolume> aux,
                                 std::shared_ptr<CudaVolume> residual)
{
    poisson_impl_->ComputeRhoAndBeta(MemPiece(beta->mem(), beta->size()),
                                     MemPiece(rho_new->mem(), rho_new->size()),
                                     MemPiece(rho->mem(), rho->size()),
                                     aux->dev_array(), residual->dev_array(),
                                     aux->size());
}

void CudaMain::ScaledAdd(std::shared_ptr<CudaVolume> dest,
                         std::shared_ptr<CudaVolume> v0,
                         std::shared_ptr<CudaVolume> v1,
                         std::shared_ptr<CudaMemPiece> coef, float sign)
{
    poisson_impl_->ScaledAdd(dest->dev_array(), v0->dev_array(),
                             v1->dev_array(),
                             MemPiece(coef->mem(), coef->size()), sign,
                             dest->size());
}

void CudaMain::ScaleVector(std::shared_ptr<CudaVolume> dest,
                           std::shared_ptr<CudaVolume> v,
                           std::shared_ptr<CudaMemPiece> coef, float sign)
{
    poisson_impl_->ScaledAdd(dest->dev_array(), nullptr, v->dev_array(),
                             MemPiece(coef->mem(), coef->size()), sign,
                             dest->size());
}

void CudaMain::AddCurlPsi(std::shared_ptr<CudaVolume> vel_x,
                          std::shared_ptr<CudaVolume> vel_y,
                          std::shared_ptr<CudaVolume> vel_z,
                          std::shared_ptr<CudaVolume> psi_x,
                          std::shared_ptr<CudaVolume> psi_y,
                          std::shared_ptr<CudaVolume> psi_z)
{
    fluid_impl_->AddCurlPsi(vel_x->dev_array(), vel_y->dev_array(),
                            vel_z->dev_array(), psi_x->dev_array(),
                            psi_y->dev_array(), psi_z->dev_array(),
                            vel_x->size());
}

void CudaMain::ApplyVorticityConfinement(std::shared_ptr<CudaVolume> vel_x,
                                         std::shared_ptr<CudaVolume> vel_y,
                                         std::shared_ptr<CudaVolume> vel_z,
                                         std::shared_ptr<CudaVolume> vort_x,
                                         std::shared_ptr<CudaVolume> vort_y,
                                         std::shared_ptr<CudaVolume> vort_z)
{
    fluid_impl_->ApplyVorticityConfinement(vel_x->dev_array(),
                                           vel_y->dev_array(),
                                           vel_z->dev_array(),
                                           vort_x->dev_array(),
                                           vort_y->dev_array(),
                                           vort_z->dev_array(),
                                           vel_x->size());
}

void CudaMain::BuildVorticityConfinement(std::shared_ptr<CudaVolume> conf_x,
                                         std::shared_ptr<CudaVolume> conf_y,
                                         std::shared_ptr<CudaVolume> conf_z,
                                         std::shared_ptr<CudaVolume> vort_x,
                                         std::shared_ptr<CudaVolume> vort_y,
                                         std::shared_ptr<CudaVolume> vort_z,
                                         float coeff)
{
    fluid_impl_->BuildVorticityConfinement(conf_x->dev_array(),
                                           conf_y->dev_array(),
                                           conf_z->dev_array(),
                                           vort_x->dev_array(),
                                           vort_y->dev_array(),
                                           vort_z->dev_array(), coeff,
                                           conf_x->size());
}

void CudaMain::ComputeCurl(std::shared_ptr<CudaVolume> vort_x,
                           std::shared_ptr<CudaVolume> vort_y,
                           std::shared_ptr<CudaVolume> vort_z,
                           std::shared_ptr<CudaVolume> vel_x,
                           std::shared_ptr<CudaVolume> vel_y,
                           std::shared_ptr<CudaVolume> vel_z)
{
    fluid_impl_->ComputeCurl(vort_x->dev_array(), vort_y->dev_array(),
                             vort_z->dev_array(), vel_x->dev_array(),
                             vel_y->dev_array(), vel_z->dev_array(),
                             vort_x->size());
}

void CudaMain::ComputeDeltaVorticity(std::shared_ptr<CudaVolume> delta_x,
                                     std::shared_ptr<CudaVolume> delta_y,
                                     std::shared_ptr<CudaVolume> delta_z,
                                     std::shared_ptr<CudaVolume> vort_x,
                                     std::shared_ptr<CudaVolume> vort_y,
                                     std::shared_ptr<CudaVolume> vort_z)
{
    fluid_impl_->ComputeDeltaVorticity(delta_x->dev_array(),
                                       delta_y->dev_array(),
                                       delta_z->dev_array(),
                                       vort_x->dev_array(), vort_y->dev_array(),
                                       vort_z->dev_array(), delta_x->size());
}

void CudaMain::DecayVortices(std::shared_ptr<CudaVolume> vort_x,
                             std::shared_ptr<CudaVolume> vort_y,
                             std::shared_ptr<CudaVolume> vort_z,
                             std::shared_ptr<CudaVolume> div, float time_step)
{
    fluid_impl_->DecayVortices(vort_x->dev_array(), vort_y->dev_array(),
                               vort_z->dev_array(), div->dev_array(), time_step,
                               vort_x->size());
}

void CudaMain::StretchVortices(std::shared_ptr<CudaVolume> vnp1_x,
                               std::shared_ptr<CudaVolume> vnp1_y,
                               std::shared_ptr<CudaVolume> vnp1_z,
                               std::shared_ptr<CudaVolume> vel_x,
                               std::shared_ptr<CudaVolume> vel_y,
                               std::shared_ptr<CudaVolume> vel_z,
                               std::shared_ptr<CudaVolume> vort_x,
                               std::shared_ptr<CudaVolume> vort_y,
                               std::shared_ptr<CudaVolume> vort_z,
                               float time_step)
{
    fluid_impl_->StretchVortices(vnp1_x->dev_array(), vnp1_y->dev_array(),
                                 vnp1_z->dev_array(), vel_x->dev_array(),
                                 vel_y->dev_array(), vel_z->dev_array(),
                                 vort_x->dev_array(), vort_y->dev_array(),
                                 vort_z->dev_array(), time_step,
                                 vnp1_x->size());
}

void CudaMain::EmitFlipParticles(FlipParticles* particles,
                                 const glm::vec3& center_point,
                                 const glm::vec3& hotspot, float radius,
                                 float density, float temperature,
                                 const glm::vec3& velocity,
                                 const glm::ivec3& volume_size)
{
    flip_impl_->Emit(ToCudaFlipParticles(*particles), center_point, hotspot,
                     radius, density, temperature, velocity, volume_size);
}

void CudaMain::MoveFlipParticles(FlipParticles* particles,
                                 int* num_active_particles,
                                 const FlipParticles* aux,
                                 std::shared_ptr<CudaVolume> vnp1_x,
                                 std::shared_ptr<CudaVolume> vnp1_y,
                                 std::shared_ptr<CudaVolume> vnp1_z,
                                 std::shared_ptr<CudaVolume> vn_x,
                                 std::shared_ptr<CudaVolume> vn_y,
                                 std::shared_ptr<CudaVolume> vn_z,
                                 std::shared_ptr<CudaVolume> density,
                                 std::shared_ptr<CudaVolume> temperature,
                                 float velocity_dissipation,
                                 float density_dissipation,
                                 float temperature_dissipation, float time_step)
{
    flip_impl_->Advect(ToCudaFlipParticles(*particles), num_active_particles,
                       ToCudaFlipParticles(*aux), vnp1_x->dev_array(),
                       vnp1_y->dev_array(), vnp1_z->dev_array(),
                       vn_x->dev_array(), vn_y->dev_array(), vn_z->dev_array(),
                       density->dev_array(), temperature->dev_array(),
                       time_step, velocity_dissipation, density_dissipation,
                       temperature_dissipation, vnp1_x->size());
}

void CudaMain::ResetFlipParticles(FlipParticles* particles,
                                  const glm::ivec3& volume_size)
{
    flip_impl_->Reset(ToCudaFlipParticles(*particles), volume_size);
}

void CudaMain::EmitParticles(std::shared_ptr<CudaLinearMemU16> pos_x,
                             std::shared_ptr<CudaLinearMemU16> pos_y,
                             std::shared_ptr<CudaLinearMemU16> pos_z,
                             std::shared_ptr<CudaLinearMemU16> density,
                             std::shared_ptr<CudaLinearMemU16> life,
                             std::shared_ptr<CudaMemPiece> tail,
                             int num_of_particles, int num_to_emit,
                             const glm::vec3& location, float radius,
                             float density_value)
{
    particle_impl_->Emit(pos_x->mem(), pos_y->mem(), pos_z->mem(),
                         density->mem(), life->mem(),
                         reinterpret_cast<int*>(tail->mem()), num_of_particles,
                         num_to_emit, location, radius, density_value);
}

void CudaMain::MoveParticles(std::shared_ptr<CudaLinearMemU16> pos_x,
                             std::shared_ptr<CudaLinearMemU16> pos_y,
                             std::shared_ptr<CudaLinearMemU16> pos_z,
                             std::shared_ptr<CudaLinearMemU16> density,
                             std::shared_ptr<CudaLinearMemU16> life,
                             int num_of_particles,
                             std::shared_ptr<CudaVolume> vel_x,
                             std::shared_ptr<CudaVolume> vel_y,
                             std::shared_ptr<CudaVolume> vel_z, float time_step)
{
    particle_impl_->Advect(pos_x->mem(), pos_y->mem(), pos_z->mem(),
                           density->mem(), life->mem(), num_of_particles,
                           vel_x->dev_array(), vel_y->dev_array(),
                           vel_z->dev_array(), time_step, vel_x->size());
}

void CudaMain::ResetParticles(std::shared_ptr<CudaLinearMemU16> life,
                              int num_of_particles)
{
    particle_impl_->Reset(life->mem(), num_of_particles);
}

bool CudaMain::CopyToVbo(uint32_t point_vbo, uint32_t extra_vbo,
                         std::shared_ptr<CudaLinearMemU16> pos_x,
                         std::shared_ptr<CudaLinearMemU16> pos_y,
                         std::shared_ptr<CudaLinearMemU16> pos_z,
                         std::shared_ptr<CudaLinearMemU16> density,
                         std::shared_ptr<CudaLinearMemU16> temperature,
                         std::shared_ptr<CudaMemPiece> num_of_actives,
                         float crit_density, int num_of_particles)
{
    auto i = registerd_buffers_.find(point_vbo);
    auto j = registerd_buffers_.find(extra_vbo);
    assert(i != registerd_buffers_.end() && j != registerd_buffers_.end());
    if (i == registerd_buffers_.end() || j == registerd_buffers_.end())
        return false;

    uint16_t* temp_field = temperature ? temperature->mem() : nullptr;
    int* count = num_of_actives ?
        reinterpret_cast<int*>(num_of_actives->mem()) : nullptr;
    core_->CopyToVbo(i->second.get(), j->second.get(), pos_x->mem(),
                     pos_y->mem(), pos_z->mem(), density->mem(),
                     temp_field, crit_density, count, num_of_particles);
    return true;
}

void CudaMain::Raycast(std::shared_ptr<GLSurface> dest,
                       std::shared_ptr<CudaVolume> density,
                       const glm::mat4& inv_rotation, const glm::vec3& eye_pos,
                       const glm::vec3& light_color, const glm::vec3& light_pos,
                       float light_intensity, float focal_length,
                       const glm::vec2& screen_size, int num_samples,
                       int num_light_samples, float absorption,
                       float density_factor, float occlusion_factor)
{
    auto i = registerd_textures_.find(dest);
    assert(i != registerd_textures_.end());
    if (i == registerd_textures_.end())
        return;

    core_->Raycast(i->second.get(), density->dev_array(), inv_rotation,
                   dest->size(), eye_pos, light_color, light_pos,
                   light_intensity, focal_length, screen_size, num_samples,
                   num_light_samples, absorption, density_factor,
                   occlusion_factor, density->size());
}

void CudaMain::SetAdvectionMethod(AdvectionMethod method)
{
    fluid_impl_->set_advect_method(ToCudaAdvectionMethod(method));
}

void CudaMain::SetMidPoint(bool mid_point)
{
    fluid_impl_->set_mid_point(mid_point);
}

void CudaMain::SetCellSize(float cell_size)
{
    fluid_impl_->set_cell_size(cell_size);
    poisson_impl_->set_cell_size(cell_size);
    flip_impl_->set_cell_size(cell_size);
    particle_impl_->set_cell_size(cell_size);
}

void CudaMain::SetFluidImpulse(FluidImpulse impulse)
{
    fluid_impl_->set_fluid_impulse(ToCudaFluidImpulse(impulse));
    flip_impl_->set_fluid_impulse(ToCudaFluidImpulse(impulse));
    particle_impl_->set_fluid_impulse(ToCudaFluidImpulse(impulse));
}

void CudaMain::SetOutflow(bool outflow)
{
    fluid_impl_->set_outflow(outflow);
    poisson_impl_->set_outflow(outflow);
    flip_impl_->set_outflow(outflow);
    particle_impl_->set_outflow(outflow);
}

void CudaMain::SetStaggered(bool staggered)
{
    fluid_impl_->set_staggered(staggered);
}

void CudaMain::ComputeResidualDiagnosis(std::shared_ptr<CudaVolume> residual,
                                        std::shared_ptr<CudaVolume> u,
                                        std::shared_ptr<CudaVolume> b)
{
    fluid_impl_->ComputeResidualDiagnosis(residual->dev_array(), u->dev_array(),
                                          b->dev_array(), residual->size());

    PrintVolume(residual, "||residual||");
}

void CudaMain::PrintVolume(std::shared_ptr<CudaVolume> volume,
                           const std::string& name)
{
    int w = volume->width();
    int h = volume->height();
    int d = volume->depth();
    int n = volume->num_of_components();
    int element_size = sizeof(float);

    static char* buf = nullptr;
    static int size[3] = {w, h, d};
    if (!buf) {
        buf = new char[w * h * d * element_size * n];
    } else {
        if (size[0] != w || size[1] != h || size[2] != d) {
            delete[] buf;

            buf = new char[w * h * d * element_size * n];
            size[0] = w;
            size[1] = h;
            size[2] = d;
        }
    }

    memset(buf, 0, w * h * d * element_size * n);
    CudaCore::CopyFromVolume(buf, w * element_size * n, volume->dev_array(),
                             volume->size());

    float* f = (float*)buf;
    double sum = 0.0;
    double q = 0.0;
    double m = 0.0;
    int c = 0;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                for (int l = 0; l < n; l++) {
                    q = f[i * w * h * n + j * w * n + k * n + l];
                    //if (i == 30 && j == 0 && k == 56)
                    //if (q > 1)
                    if (q > 0.0f) {
                        sum += q;
                        c++;
                    }

                    m = std::max(q, m);
                }
            }
        }
    }

    c = std::max(1, c);
    double avg = sum / c;
    PrintDebugString("(CUDA) avg %s: %.8f,    max %s: %.8f\n", name.c_str(),
                     avg, name.c_str(), m);
}

void CudaMain::RoundPassed(int round)
{
    fluid_impl_->RoundPassed(round);
}

void CudaMain::Sync()
{
    core_->Sync();
}
