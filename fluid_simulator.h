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
    void set_solver_choice(PoissonMethod method) { solver_choice_ = method; }
    void set_num_multigrid_iterations(int n) { num_multigrid_iterations_ = n; }
    void set_num_full_multigrid_iterations(int n) {
        num_full_multigrid_iterations_ = n;
    }

    void Advect(std::shared_ptr<GLTexture> velocity,
                std::shared_ptr<GLTexture> source,
                std::shared_ptr<GLTexture> dest, float delta_time,
                float dissipation);
    void AdvectVelocity(std::shared_ptr<GLTexture> velocity,
                        std::shared_ptr<GLTexture> dest, float delta_time,
                        float dissipation);
    void Jacobi(std::shared_ptr<GLTexture> pressure,
                std::shared_ptr<GLTexture> divergence);
    void DampedJacobi(std::shared_ptr<GLTexture> pressure,
                      std::shared_ptr<GLTexture> divergence, float cell_size);
    void SolvePressure(std::shared_ptr<GLTexture> packed);
    void SubtractGradient(std::shared_ptr<GLTexture> velocity,
                          std::shared_ptr<GLTexture> packed);
    void ComputeDivergence(std::shared_ptr<GLTexture> velocity,
                           std::shared_ptr<GLTexture> dest);
    void ApplyImpulse(std::shared_ptr<GLTexture> dest,
                      Vectormath::Aos::Vector3 position,
                      Vectormath::Aos::Vector3 hotspot, float value);
    void ApplyBuoyancy(std::shared_ptr<GLTexture> velocity,
                       std::shared_ptr<GLTexture> temperature,
                       std::shared_ptr<GLTexture> dest, float delta_time);

private:
    PoissonMethod solver_choice_;
    int num_multigrid_iterations_;
    int num_full_multigrid_iterations_;
};

#endif // _FLUID_SIMULATOR_H_