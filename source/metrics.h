//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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

        FLIP_EMISSION,
        FLIP_INTERPOLATION,
        FLIP_RESAMPLING,
        FLIP_ADVECTION,
        FLIP_CELL_BINDING,
        FLIP_PREFIX_SUM,
        FLIP_SORTING,
        FLIP_TRANSFER,

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

    void OnParticleEmitted();
    void OnParticleVelocityInterpolated();
    void OnParticleResampled();
    void OnParticleAdvected();
    void OnParticleCellBound();
    void OnParticlePrefixSumCalculated();
    void OnParticleSorted();
    void OnParticleTransferred();

    void OnVorticityRestored();
    void OnRaycastPerformed();

    void OnParticleNumberUpdated(int n);
    void OnProlongated();

    int GetActiveParticleNumber() const;
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
    int num_active_particles_;
};

#endif // _METRICS_H_