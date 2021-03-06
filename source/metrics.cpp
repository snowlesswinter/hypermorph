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
#include "metrics.h"

#include <numeric>

namespace
{
const size_t kMaxNumOfTimeStamps = 100;
const int kNumOfSamples = 20;
}

Metrics* Metrics::Instance()
{
    static Metrics* m = nullptr;
    if (!m)
        m = new Metrics();

    return m;
}

Metrics::Metrics()
    : diagnosis_mode_(false)
    , sync_operation_()
    , get_time_()
    , time_stamps_()
    , last_operation_time_(0.0)
    , operation_time_costs_()
    , num_active_particles_(0)
{
}

Metrics::~Metrics()
{
}

void Metrics::set_diagnosis_mode(bool diagnosis)
{
    diagnosis_mode_ = diagnosis;
}

void Metrics::SetOperationSync(const std::function<void (void)>& operation_sync)
{
    sync_operation_ = operation_sync;
}

void Metrics::SetTimeSource(const std::function<double (void)>& time_source)
{
    get_time_ = time_source;
}

void Metrics::OnFrameRendered()
{
    if (!get_time_)
        return;

    time_stamps_.push_front(get_time_());
    while (time_stamps_.size() > kMaxNumOfTimeStamps)
        time_stamps_.pop_back();

    OnOperationProceeded(RENDER_DENSITY);
}

float Metrics::GetFrameRate() const
{
    if (time_stamps_.size() <= 1)
        return 0.0f;

    return static_cast<float>(
        time_stamps_.size() / (time_stamps_.front() - time_stamps_.back()));
}

void Metrics::OnFrameUpdateBegins()
{
    if (!diagnosis_mode_ || !get_time_)
        return;

    if (sync_operation_)
        sync_operation_();

    last_operation_time_ = get_time_();
}

void Metrics::OnFrameRenderingBegins()
{
    if (!diagnosis_mode_ || !get_time_)
        return;

    if (sync_operation_)
        sync_operation_();

    last_operation_time_ = get_time_();
}

void Metrics::OnVelocityAvected()
{
    OnOperationProceeded(AVECT_VELOCITY);
}

void Metrics::OnTemperatureAvected()
{
    OnOperationProceeded(AVECT_TEMPERATURE);
}

void Metrics::OnDensityAvected()
{
    OnOperationProceeded(AVECT_DENSITY);
}

void Metrics::OnBuoyancyApplied()
{
    OnOperationProceeded(APPLY_BUOYANCY);
}

void Metrics::OnImpulseApplied()
{
    OnOperationProceeded(APPLY_IMPULSE);
}

void Metrics::OnDivergenceComputed()
{
    OnOperationProceeded(COMPUTE_DIVERGENCE);
}

void Metrics::OnPressureSolved()
{
    OnOperationProceeded(SOLVE_PRESSURE);
}

void Metrics::OnVelocityRectified()
{
    OnOperationProceeded(RECTIFY_VELOCITY);
}

void Metrics::OnParticleEmitted()
{
    OnOperationProceeded(FLIP_EMISSION);
}

void Metrics::OnParticleVelocityInterpolated()
{
    OnOperationProceeded(FLIP_INTERPOLATION);
}

void Metrics::OnParticleResampled()
{
    OnOperationProceeded(FLIP_RESAMPLING);
}

void Metrics::OnParticleAdvected()
{
    OnOperationProceeded(FLIP_ADVECTION);
}

void Metrics::OnParticleCellBound()
{
    OnOperationProceeded(FLIP_CELL_BINDING);
}

void Metrics::OnParticlePrefixSumCalculated()
{
    OnOperationProceeded(FLIP_PREFIX_SUM);
}

void Metrics::OnParticleSorted()
{
    OnOperationProceeded(FLIP_SORTING);
}

void Metrics::OnParticleTransferred()
{
    OnOperationProceeded(FLIP_TRANSFER);
}

void Metrics::OnVorticityRestored()
{
    OnOperationProceeded(RESTORE_VORTICITY);
}

void Metrics::OnRaycastPerformed()
{
    OnOperationProceeded(PERFORM_RAYCAST);
}

void Metrics::OnParticleNumberUpdated(int n)
{
    num_active_particles_ = n;
}

void Metrics::OnProlongated()
{
    OnOperationProceeded(POISSON_PROLONGATE);
}

int Metrics::GetActiveParticleNumber() const
{
    return num_active_particles_;
}

float Metrics::GetOperationTimeCost(Operations o) const
{
    auto& samples = operation_time_costs_[o];
    if (samples.empty())
        return 0.0f;

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return static_cast<float>(sum / samples.size());
}

void Metrics::Reset()
{
    time_stamps_.clear();
    last_operation_time_ = 0.0;
    num_active_particles_ = 0;
    for (auto& i : operation_time_costs_)
        i.clear();
}

void Metrics::OnOperationProceeded(Operations o)
{
    if (!diagnosis_mode_ || !get_time_)
        return;

    if (sync_operation_)
        sync_operation_();

    auto& samples = operation_time_costs_[o];
    double current_time = get_time_();

    // Store in microseconds.
    samples.push_front((current_time - last_operation_time_) * 1000000.0);
    while (samples.size() > kNumOfSamples)
        samples.pop_back();

    last_operation_time_ = current_time;
}
