#include "stdafx.h"
#include "fluid_unittest.h"

#include "cuda_host/cuda_volume.h"
#include "fluid_simulator.h"
#include "graphics_volume.h"
#include "half_float/half.h"
#include "opengl/gl_volume.h"
#include "third_party/glm/vec3.hpp"
#include "unittest_common.h"
#include "utility.h"

namespace
{
const float kTimeStep = 0.33f;

void InitializeDensityVolume(GraphicsVolume* cuda_volume,
                             GraphicsVolume* glsl_volume, int size,
                             const std::pair<float, float>& scope)
{
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    for (auto& i : test_data)
        i = half(UnittestCommon::RandomFloat(scope)).bits();

    // Volumes that registered to CUDA can not be fed??
    cuda_volume->gl_volume()->SetTexImage(&test_data[0]);
    glsl_volume->gl_volume()->SetTexImage(&test_data[0]);
}
} // Anonymous namespace.

void FluidUnittest::TestBuoyancyApplication(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n_4 = 4;
    int n_1 = 1;
    int pitch_4 = width * sizeof(uint16_t) * n_4;
    int pitch_1 = width * sizeof(uint16_t) * n_1;
    int size_4 = pitch_4 * height * depth;
    int size_1 = pitch_1 * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume4(sim_cuda.velocity_.get(),
                                      sim_glsl.velocity_.get(), width, height,
                                      depth, n_4, pitch_4, size_4,
                                      std::make_pair(-5.0f, 5.0f));
    UnittestCommon::InitializeVolume1(sim_cuda.temperature_.get(),
                                      sim_glsl.temperature_.get(), width,
                                      height, depth, n_1, pitch_1, size_1,
                                      std::make_pair(0.0f, 40.0f));

    sim_cuda.ApplyBuoyancy(kTimeStep);
    sim_glsl.ApplyBuoyancy(kTimeStep);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size_4,
                                           pitch_4, n_4, 7,
                                           sim_cuda.velocity_.get(),
                                           sim_glsl.velocity_.get(),
                                           __FUNCTION__);
}

void FluidUnittest::TestDampedJacobi(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.packed_->GetWidth();
    int height = sim_cuda.packed_->GetHeight();
    int depth = sim_cuda.packed_->GetDepth();
    int n = 2;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume2(sim_cuda.packed_.get(),
                                      sim_glsl.packed_.get(), width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-5.0f, 5.0f));

    sim_cuda.DampedJacobi(CellSize, 1);
    sim_glsl.DampedJacobi(CellSize, 1);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           1, sim_cuda.packed_.get(),
                                           sim_glsl.packed_.get(),
                                           __FUNCTION__);
}

void FluidUnittest::TestDensityAdvection(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n_v = 4;
    int n_d = 1;
    int pitch_v = width * sizeof(uint16_t) * n_v;
    int pitch_d = width * sizeof(uint16_t) * n_d;
    int size_v = pitch_v * height * depth;
    int size_d = pitch_d * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume4(sim_cuda.velocity_.get(),
                                      sim_glsl.velocity_.get(), width, height,
                                      depth, n_v, pitch_v, size_v,
                                      std::make_pair(-5.0f, 5.0f));
    InitializeDensityVolume(sim_cuda.density_.get(), sim_glsl.density_.get(),
                            size_d, std::make_pair(0.0f, 3.0f));

    sim_cuda.AdvectDensity(kTimeStep);
    sim_glsl.AdvectDensity(kTimeStep);

    // Copy the result back to CPU.
    glm::ivec3 volume_size(width, height, depth);
    std::vector<uint16_t> result_cuda(size_d, 0);
    sim_cuda.density_->gl_volume()->GetTexImage(&result_cuda[0]);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_glsl(size_d, 0);
    sim_glsl.density_->gl_volume()->GetTexImage(&result_glsl[0]);

    UnittestCommon::VerifyResult1(result_cuda, result_glsl, width, height,
                                  depth, n_d, __FUNCTION__);
}

void FluidUnittest::TestDivergenceCalculation(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume4(sim_cuda.velocity_.get(),
                                      sim_glsl.velocity_.get(), width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-5.0f, 5.0f));

    sim_cuda.ComputeDivergence();
    sim_glsl.ComputeDivergence();

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           2, sim_cuda.general4_.get(),
                                           sim_glsl.general4_.get(),
                                           __FUNCTION__);
}

void FluidUnittest::TestGradientSubtraction(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume4(sim_cuda.velocity_.get(),
                                      sim_glsl.velocity_.get(), width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-5.0f, 5.0f));
    UnittestCommon::InitializeVolume4(sim_cuda.general4_.get(),
                                      sim_glsl.general4_.get(), width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    sim_cuda.SubtractGradient();
    sim_glsl.SubtractGradient();

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           7, sim_cuda.velocity_.get(),
                                           sim_glsl.velocity_.get(),
                                           __FUNCTION__);
}

void FluidUnittest::TestTemperatureAdvection(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n_4 = 4;
    int n_1 = 1;
    int pitch_4 = width * sizeof(uint16_t) * n_4;
    int pitch_1 = width * sizeof(uint16_t) * n_1;
    int size_4 = pitch_4 * height * depth;
    int size_1 = pitch_1 * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume4(sim_cuda.velocity_.get(),
                                      sim_glsl.velocity_.get(), width, height,
                                      depth, n_4, pitch_4, size_4,
                                      std::make_pair(-5.0f, 5.0f));
    UnittestCommon::InitializeVolume1(sim_cuda.temperature_.get(),
                                      sim_glsl.temperature_.get(), width,
                                      height, depth, n_1, pitch_1, size_1,
                                      std::make_pair(0.0f, 40.0f));

    sim_cuda.AdvectTemperature(kTimeStep);
    sim_glsl.AdvectTemperature(kTimeStep);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size_1,
                                           pitch_1, n_1, 1,
                                           sim_cuda.temperature_.get(),
                                           sim_glsl.temperature_.get(),
                                           __FUNCTION__);
}

void FluidUnittest::TestVelocityAdvection(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!UnittestCommon::InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;

    // Copy the initialized data to GPU.
    UnittestCommon::InitializeVolume4(sim_cuda.velocity_.get(),
                                      sim_glsl.velocity_.get(), width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-5.0f, 5.0f));

    sim_cuda.AdvectVelocity(kTimeStep);
    sim_glsl.AdvectVelocity(kTimeStep);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           7, sim_cuda.velocity_.get(),
                                           sim_glsl.velocity_.get(),
                                           __FUNCTION__);
}
