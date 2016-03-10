#ifndef _METRICS_H_
#define _METRICS_H_

#include <list>

class Metrics
{
public:
    Metrics();
    ~Metrics();

    void OnFrameRendered(float current_time);
    float GetFrameRate(float current_time) const;

private:
    std::list<float> time_stamps_;
};

#endif // _METRICS_H_