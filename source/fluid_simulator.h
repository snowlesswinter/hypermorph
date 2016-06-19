#ifndef _FLUID_SIMULATOR_H_
#define _FLUID_SIMULATOR_H_

#include <memory>

#include "graphics_lib_enum.h"
#include "graphics_volume_group.h"
#include "third_party/glm/vec3.hpp"

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
        POISSON_SOLVER_FULL_MULTI_GRID,
        POISSON_SOLVER_MULTI_GRID_PRECONDITIONED_CONJUGATE_GRADIENT
    };

    FluidSimulator();
    ~FluidSimulator();

    bool Init();
    void Reset();
    std::shared_ptr<GraphicsVolume> GetDensityField() const;
    bool IsImpulsing() const;
    void NotifyConfigChanged();
    void StartImpulsing(float x, float y);
    void StopImpulsing();
    void Update(float delta_time, double seconds_elapsed, int frame_count);
    void UpdateImpulsing(float x, float y);

    void set_solver_choice(PoissonMethod method) { solver_choice_ = method; }
    void set_diagnosis(int diagnosis);
    GraphicsLib graphics_lib() const { return graphics_lib_; }
    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }

private:
    friend class FluidUnittest;

    void AdvectDensity(float cell_size, float delta_time);
    void AdvectImpl(std::shared_ptr<GraphicsVolume> source, float delta_time,
                    float dissipation);
    void AdvectTemperature(float cell_size, float delta_time);
    void AdvectVelocity(float cell_size, float delta_time);
    void ApplyBuoyancy(float delta_time);
    void ApplyImpulse(double seconds_elapsed, float delta_time);
    void ComputeDivergence(std::shared_ptr<GraphicsVolume> divergence,
                           float cell_size);
    void ComputeResidualDiagnosis(float cell_size);
    void DampedJacobi(std::shared_ptr<GraphicsVolume> pressure,
                      std::shared_ptr<GraphicsVolume> divergence,
                      float cell_size, int num_of_iterations);
    void Impulse(std::shared_ptr<GraphicsVolume> dest,
                 const glm::vec3& position, const glm::vec3& hotspot,
                 float splat_radius, float value);
    void ImpulseDensity(const glm::vec3& position, const glm::vec3& hotspot,
                        float splat_radius,
                        float value);
    void ReviseDensity();
    void SolvePressure(std::shared_ptr<GraphicsVolume> pressure,
                       std::shared_ptr<GraphicsVolume> divergence,
                       float cell_size);
    void SubtractGradient(std::shared_ptr<GraphicsVolume> pressure,
                          float cell_size);

    // Vorticity.
    void AddCurlPsi(const GraphicsVolume3& psi, float cell_size);
    void AdvectVortices(const GraphicsVolume3& vorticity,
                        const GraphicsVolume3& temp,
                        std::shared_ptr<GraphicsVolume> aux, float cell_size,
                        float delta_time);
    void ApplyVorticityConfinemnet();
    void BuildVorticityConfinemnet(float delta_time, float cell_size);
    void ComputeCurl(const GraphicsVolume3& vorticity,
                     const GraphicsVolume3& velocity, float cell_size);
    void ComputeDeltaVorticity(const GraphicsVolume3& aux,
                               const GraphicsVolume3& vorticity);
    void DecayVortices(const GraphicsVolume3& vorticity,
                       std::shared_ptr<GraphicsVolume> aux, float delta_time,
                       float cell_size);
    void RestoreVorticity(float delta_time, float cell_size);
    void SolvePsi(const GraphicsVolume3& psi,
                  const GraphicsVolume3& delta_vort, float cell_size);
    void StretchVortices(const GraphicsVolume3& vort_np1,
                         const GraphicsVolume3& vorticity, float delta_time,
                         float cell_size);

    const GraphicsVolume3& GetVorticityField();
    const GraphicsVolume3& GetAuxField();
    const GraphicsVolume3& GetVorticityConfinementField();

    glm::vec3 grid_size_;
    float cell_size_;
    GraphicsLib graphics_lib_;
    PoissonMethod solver_choice_;
    std::unique_ptr<MultigridCore> multigrid_core_;
    std::unique_ptr<PoissonSolver> pressure_solver_;
    std::unique_ptr<PoissonSolver> psi_solver_;
    int volume_byte_width_;
    int diagnosis_;

    GraphicsVolume3 velocity_;
    GraphicsVolume3 velocity_prime_;
    GraphicsVolume3 vorticity_;
    GraphicsVolume3 aux_;
    GraphicsVolume3 vort_conf_;
    std::shared_ptr<GraphicsVolume> density_;
    std::shared_ptr<GraphicsVolume> temperature_;
    std::shared_ptr<GraphicsVolume> general1a_;
    std::shared_ptr<GraphicsVolume> general1b_;
    std::shared_ptr<GraphicsVolume> general1c_;
    std::shared_ptr<GraphicsVolume> general1d_;
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;

    std::shared_ptr<glm::vec2> manual_impulse_;
};

#endif // _FLUID_SIMULATOR_H_