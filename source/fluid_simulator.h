#ifndef _FLUID_SIMULATOR_H_
#define _FLUID_SIMULATOR_H_

#include <memory>

#include "graphics_lib_enum.h"
#include "graphics_volume_group.h"
#include "third_party/glm/fwd.hpp"

class FluidUnittest;
class GraphicsVolume;
class MultigridCore;
class OpenBoundaryMultigridPoissonSolver;
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
    void SetStaggered(bool staggered);
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
    void ComputeCurl(const GraphicsVolume3* vorticity,
                     std::shared_ptr<GraphicsVolume> velocity);
    void ComputeDivergence();
    void ComputeResidualDiagnosis(float cell_size);
    void DampedJacobi(float cell_size, int num_of_iterations);
    void Impulse(std::shared_ptr<GraphicsVolume> dest,
                 const glm::vec3& position, const glm::vec3& hotspot,
                 float splat_radius, const glm::vec3& value, uint32_t mask);
    void ImpulseDensity(const glm::vec3& position, const glm::vec3& hotspot,
                        float splat_radius,
                        float value);
    void ReviseDensity();
    void SolvePressure();
    void SubtractGradient();

    // Vorticity.
    void AddCurlPsi();
    void AdvectVortices(float delta_time);
    void ApplyVorticityConfinemnet();
    void BuildVorticityConfinemnet(float delta_time);
    void ComputeDeltaVorticity();
    void DecayVortices(float delta_time, float cell_size);
    void RestoreVorticity(float delta_time);
    void SolvePsi();
    void StretchVortices(float delta_time, float cell_size);

    const GraphicsVolume3& GetVorticityField();
    const GraphicsVolume3& GetAuxField();
    const GraphicsVolume3& GetVorticityConfinementField();

    GraphicsLib graphics_lib_;
    PoissonMethod solver_choice_;
    std::unique_ptr<MultigridCore> multigrid_core_;
    std::unique_ptr<PoissonSolver> pressure_solver_;
    std::unique_ptr<OpenBoundaryMultigridPoissonSolver> psi_solver_;
    int num_multigrid_iterations_;
    int num_full_multigrid_iterations_;
    int volume_byte_width_;
    bool diagnosis_;

    GraphicsVolume3 velocity_;
    GraphicsVolume3 vorticity_;
    GraphicsVolume3 aux_;
    GraphicsVolume3 vort_conf_;
    std::shared_ptr<GraphicsVolume> density_;
    std::shared_ptr<GraphicsVolume> temperature_;
    std::shared_ptr<GraphicsVolume> pressure_;
    std::shared_ptr<GraphicsVolume> general1a_;
    std::shared_ptr<GraphicsVolume> general1b_;
    std::shared_ptr<GraphicsVolume> general1c_;
    std::shared_ptr<GraphicsVolume> general1d_;
    std::shared_ptr<GraphicsVolume> general1e_;
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;

    std::shared_ptr<glm::vec2> manual_impulse_;
};

#endif // _FLUID_SIMULATOR_H_