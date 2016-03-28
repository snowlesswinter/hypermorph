#ifndef _FLUID_SIMULATOR_H_
#define _FLUID_SIMULATOR_H_

#include <memory>

#include "graphics_lib_enum.h"

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class GraphicsVolume;
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
    void Update(float delta_time, double seconds_elapsed, int frame_count);

    void set_solver_choice(PoissonMethod method) { solver_choice_ = method; }
    void set_num_multigrid_iterations(int n) { num_multigrid_iterations_ = n; }
    void set_num_full_multigrid_iterations(int n) {
        num_full_multigrid_iterations_ = n;
    }
    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }

    // TODO
    const GraphicsVolume& GetDensityTexture() const;

private:
    void AdvectDensity(float delta_time);
    void AdvectImpl(std::shared_ptr<GraphicsVolume> source, float delta_time,
                    float dissipation);
    void AdvectTemperature(float delta_time);
    void AdvectVelocity(float delta_time);
    void ApplyBuoyancy(float delta_time);
    void ApplyImpulse(std::shared_ptr<GraphicsVolume>* dest,
                      Vectormath::Aos::Vector3 position,
                      Vectormath::Aos::Vector3 hotspot, float value);
    void ApplyImpulseDensity(Vectormath::Aos::Vector3 position,
                             Vectormath::Aos::Vector3 hotspot, float value);
    void ComputeDivergence();
    void DampedJacobi(float cell_size);
    void Jacobi(float cell_size);
    void SolvePressure();
    void SubtractGradient();

    PoissonMethod solver_choice_;
    int num_multigrid_iterations_;
    int num_full_multigrid_iterations_;

    std::shared_ptr<GraphicsVolume> velocity_;
    std::shared_ptr<GraphicsVolume> density_;
    std::shared_ptr<GraphicsVolume> density2_;
    std::shared_ptr<GraphicsVolume> temperature_;
    std::shared_ptr<GraphicsVolume> general1_;
    std::shared_ptr<GraphicsVolume> general4_;

//     std::shared_ptr<CudaVolume> velocity_cuda_;
//     std::shared_ptr<CudaVolume> temperature_cuda_;
//     std::shared_ptr<CudaVolume> general1_cuda_;
//     std::shared_ptr<CudaVolume> general4_cuda_;

    GraphicsLib graphics_lib_;
};

#endif // _FLUID_SIMULATOR_H_