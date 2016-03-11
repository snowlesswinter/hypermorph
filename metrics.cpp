#include "stdafx.h"
#include "metrics.h"

#include <numeric>

namespace
{
const size_t kMaxNumOfTimeStamps = 500;
const int kNumOfSamples = 20;
}

Metrics::Metrics()
    : time_stamps_()
    , last_operation_time_(0.0)
    , operation_time_costs_()
{
}

Metrics::~Metrics()
{
}

void Metrics::OnFrameRendered(double current_time)
{
    time_stamps_.push_front(current_time);
    while (time_stamps_.size() > kMaxNumOfTimeStamps)
        time_stamps_.pop_back();

    OnOperationProceeded(PERFORM_RAYCAST, current_time);
}

float Metrics::GetFrameRate() const
{
    if (time_stamps_.size() <= 1)
        return 0.0f;

    return static_cast<float>(
        time_stamps_.size() / (time_stamps_.front() - time_stamps_.back()));
}

void Metrics::OnFrameUpdateBegins(double current_time)
{
    last_operation_time_ = current_time;
}

void Metrics::OnFrameRenderingBegins(double current_time)
{
    last_operation_time_ = current_time;
}

void Metrics::OnVelocityAvected(double current_time)
{
    OnOperationProceeded(AVECT_VELOCITY, current_time);
}

void Metrics::OnTemperatureAvected(double current_time)
{
    OnOperationProceeded(AVECT_TEMPERATURE, current_time);
}

void Metrics::OnDensityAvected(double current_time)
{
    OnOperationProceeded(AVECT_DENSITY, current_time);
}

void Metrics::OnBuoyancyApplied(double current_time)
{
    OnOperationProceeded(APPLY_BUOYANCY, current_time);
}

void Metrics::OnImpulseApplied(double current_time)
{
    OnOperationProceeded(APPLY_IMPULSE, current_time);
}

void Metrics::OnDivergenceComputed(double current_time)
{
    OnOperationProceeded(COMPUTE_DIVERGENCE, current_time);
}

void Metrics::OnPressureSolved(double current_time)
{
    OnOperationProceeded(SOLVE_PRESSURE, current_time);
}

void Metrics::OnVelocityRectified(double current_time)
{
    OnOperationProceeded(RECTIFY_VELOCITY, current_time);
}

float Metrics::GetOperationTimeCost(Operations o) const
{
    auto& samples = operation_time_costs_[o];
    if (samples.empty())
        return 0.0f;

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return static_cast<float>(sum / samples.size());
}

void Metrics::OnOperationProceeded(Operations o, double current_time)
{
    auto& samples = operation_time_costs_[o];

    // Store in microseconds.
    samples.push_front((current_time - last_operation_time_) * 1000000.0);
    while (samples.size() > kNumOfSamples)
        samples.pop_back();

    last_operation_time_ = current_time;
}