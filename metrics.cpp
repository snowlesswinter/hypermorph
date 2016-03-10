#include "stdafx.h"
#include "metrics.h"

namespace
{
const size_t kMaxNumOfTimeStamps = 500;
}

Metrics::Metrics()
    : time_stamps_()
{
}

Metrics::~Metrics()
{
}

void Metrics::OnFrameRendered(float current_time)
{
    time_stamps_.push_front(current_time);
    while (time_stamps_.size() > kMaxNumOfTimeStamps)
        time_stamps_.pop_back();
}

float Metrics::GetFrameRate(float current_time) const
{
    if (time_stamps_.empty())
        return 0.0f;

    return time_stamps_.size() / (current_time - time_stamps_.back());
}
