#ifndef _METRICS_H_
#define _METRICS_H_

#include <array>
#include <functional>
#include <list>

class Metrics
{
public:
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
        RESTORE_VORTICITY,
        PERFORM_RAYCAST,
        RENDER_DENSITY,

        POISSON_PROLONGATE,

        NUM_OF_OPERATIONS
    };

    static Metrics* Instance();

    Metrics();
    ~Metrics();

    bool diagnosis_mode() const { return diagnosis_mode_; }
    void set_diagnosis_mode(bool diagnosis);
    void SetOperationSync(const std::function<void (void)>& operation_sync);
    void SetTimeSource(const std::function<double (void)>& time_source);

    void OnFrameRendered();
    float GetFrameRate() const;

    void OnFrameUpdateBegins();
    void OnFrameRenderingBegins();
    void OnVelocityAvected();
    void OnTemperatureAvected();
    void OnDensityAvected();
    void OnBuoyancyApplied();
    void OnImpulseApplied();
    void OnDivergenceComputed();
    void OnPressureSolved();
    void OnVelocityRectified();
    void OnVorticityRestored();
    void OnRaycastPerformed();

    void OnProlongated();

    float GetOperationTimeCost(Operations o) const;

    void Reset();

private:
    typedef std::array<std::list<double>, NUM_OF_OPERATIONS> SampleArray;

    void OnOperationProceeded(Operations o);

    bool diagnosis_mode_;
    std::function<void (void)> sync_operation_;
    std::function<double (void)> get_time_;
    std::list<double> time_stamps_;
    double last_operation_time_;
    SampleArray operation_time_costs_;
};

#endif // _METRICS_H_