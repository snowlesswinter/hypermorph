#ifndef _FLUID_SIMULATOR_H_
#define _FLUID_SIMULATOR_H_

#include <memory>

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class GLTexture;
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
    void set_use_cuda(bool use_cuda) { use_cuda_ = use_cuda; }

    // TODO
    const GLTexture& GetDensityTexture() const;

private:
    void AdvectDensity(float delta_time);
    void AdvectImpl(std::shared_ptr<GLTexture> source, float delta_time,
                    float dissipation);
    void AdvectTemperature(float delta_time);
    void AdvectVelocity(float delta_time);
    void Jacobi(std::shared_ptr<GLTexture> pressure,
                std::shared_ptr<GLTexture> divergence);
    void DampedJacobi(std::shared_ptr<GLTexture> pressure,
                      std::shared_ptr<GLTexture> divergence, float cell_size);
    void SolvePressure();
    void SubtractGradient();
    void ComputeDivergence();
    void ApplyImpulse(std::shared_ptr<GLTexture> dest,
                      Vectormath::Aos::Vector3 position,
                      Vectormath::Aos::Vector3 hotspot, float value);
    void ApplyBuoyancy(float delta_time);

    PoissonMethod solver_choice_;
    int num_multigrid_iterations_;
    int num_full_multigrid_iterations_;

    std::shared_ptr<GLTexture> velocity_;
    std::shared_ptr<GLTexture> density_;
    std::shared_ptr<GLTexture> temperature_;
    std::shared_ptr<GLTexture> general1_;
    std::shared_ptr<GLTexture> general3_;

    // TODO
    bool use_cuda_;
};

#endif // _FLUID_SIMULATOR_H_