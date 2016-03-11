#ifndef _METRICS_H_
#define _METRICS_H_

#include <array>
#include <list>

class Metrics
{
public:
    Metrics();
    ~Metrics();

    void OnFrameRendered(float current_time);
    float GetFrameRate(float current_time) const;

    void OnFrameBegins(double current_time);
    void OnVelocityAvected(double current_time);
    void OnTemperatureAvected(double current_time);
    void OnDensityAvected(double current_time);
    void OnBuoyancyApplied(double current_time);
    void OnImpulseApplied(double current_time);
    void OnDivergenceComputed(double current_time);
    void OnPressureSolved(double current_time);
    void OnVelocityRectified(double current_time);

private:
    enum Operations
    {
        AVECT_VELOCITY,
        AVECT_TEMPERATURE,
        AVECT_DENSITY,
        APPLY_BUOYANCY,
        APPLY_IMPULSE,
        COMPUTE_DIVERGENCE,
        SOLVE_PRESSURE,
        RECTIFY_VELOCITY,

        NUM_OF_OPERATIONS
    };

    typedef std::array<std::list<double>, NUM_OF_OPERATIONS> SampleArray;

    void OnOperationProceeded(Operations o, double current_time);

    std::list<float> time_stamps_;
    double last_operation_time_;
    SampleArray operation_time_costs_;
};

#endif // _METRICS_H_