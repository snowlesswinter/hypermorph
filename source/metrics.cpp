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

    OnOperationProceeded(PERFORM_RAYCAST);
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

void Metrics::OnProlongated()
{
    OnOperationProceeded(POISSON_PROLONGATE);
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