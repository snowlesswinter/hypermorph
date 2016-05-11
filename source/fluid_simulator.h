#ifndef _FLUID_SIMULATOR_H_
#define _FLUID_SIMULATOR_H_

#include <memory>

#include "graphics_lib_enum.h"
#include "third_party/glm/fwd.hpp"

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class FluidUnittest;
class GraphicsVolume;
class MultigridCore;
class PoissonSolver;
class FluidSimulator
{
public:
    enum PoissonMethod
    {
        POISSON_SOLVER_JACOBI,
        POISSON_SOLVER_DAMPED_JACOBI,
        POISSON_SOLVER_GAUSS_SEIDEL,
        POISSON_SOLVER_MULTI_GRID,
        POISSON_SOLVER_FULL_MULTI_GRID
    };

    FluidSimulator();
    ~FluidSimulator();

    bool Init();
    void Reset();
    std::shared_ptr<GraphicsVolume> GetDensityField() const;
    void StartImpulsing(float x, float y);
    void StopImpulsing();
    void Update(float delta_time, double seconds_elapsed, int frame_count);
    void UpdateImpulsing(float x, float y);

    void set_solver_choice(PoissonMethod method) { solver_choice_ = method; }
    void set_num_multigrid_iterations(int n) { num_multigrid_iterations_ = n; }
    void set_num_full_multigrid_iterations(int n) {
        num_full_multigrid_iterations_ = n;
    }
    void set_diagnosis(bool diagnosis) { diagnosis_ = diagnosis; }
    GraphicsLib graphics_lib() const { return graphics_lib_; }
    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }

private:
    friend class FluidUnittest;

    void AdvectDensity(float delta_time);
    void AdvectImpl(std::shared_ptr<GraphicsVolume> source, float delta_time,
                    float dissipation);
    void AdvectTemperature(float delta_time);
    void AdvectVelocity(float delta_time);
    void ApplyBuoyancy(float delta_time);
    void ApplyImpulse(double seconds_elapsed, float delta_time);
    void ComputeDivergence();
    void ComputeResidualDiagnosis(float cell_size);
    void DampedJacobi(float cell_size, int num_of_iterations);
    void Impulse(std::shared_ptr<GraphicsVolume> dest,
                 const glm::vec3& position, const glm::vec3& hotspot,
                 float splat_radius, const glm::vec3& value, uint32_t mask);
    void ImpulseDensity(const glm::vec3& position, const glm::vec3& hotspot,
                        float splat_radius,
                        float value);
    void Jacobi(float cell_size);
    void ReviseDensity();
    void SolvePressure();
    void SubtractGradient();

    GraphicsLib graphics_lib_;
    PoissonMethod solver_choice_;
    std::unique_ptr<MultigridCore> multigrid_core_;
    std::unique_ptr<PoissonSolver> solver_;
    int num_multigrid_iterations_;
    int num_full_multigrid_iterations_;
    int volume_byte_width_;
    bool diagnosis_;

    std::shared_ptr<GraphicsVolume> velocity_;
    std::shared_ptr<GraphicsVolume> density_;
    std::shared_ptr<GraphicsVolume> density2_;
    std::shared_ptr<GraphicsVolume> temperature_;
    std::shared_ptr<GraphicsVolume> packed_;
    std::shared_ptr<GraphicsVolume> general1_;
    std::shared_ptr<GraphicsVolume> general4_;
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;

    std::shared_ptr<glm::vec2> manual_impulse_;
};

#endif // _FLUID_SIMULATOR_H_