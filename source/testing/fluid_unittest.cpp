#include "stdafx.h"
#include "fluid_unittest.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "cuda/cuda_core.h"
#include "cuda_host/cuda_volume.h"
#include "fluid_simulator.h"
#include "graphics_volume.h"
#include "half_float/half.h"
#include "opengl/gl_texture.h"
#include "opengl/gl_utility.h"
#include "utility.h"
#include "vmath.hpp"

namespace
{
const float kTimeStep = 0.33f;

float random()
{
    int l = 10000;
    double r = static_cast<double>(rand() % l - (l >> 1)) / l;
    return static_cast<float>(r) * 10.0f;
}

void VerifyResult1(const std::vector<uint16_t>& result_cuda,
                   const std::vector<uint16_t>& result_glsl,
                   int width, int height, int depth, int n, char* function_name)
{
    assert(n == 1);

    // Compute max |e| and avg |e|.
    double max_error = 0.0;
    double sum_error = 0.0;
    double avg_error = 0.0;
    int count = 0;
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                for (int l = 0; l < n; l += n) {
                    int index =
                        i * width * height * n + j * width * n + k * n + l;

                    half h_cuda0; h_cuda0.setBits(result_cuda[index]);
                    half h_glsl0; h_glsl0.setBits(result_glsl[index]);

                    float v_cuda0 = h_cuda0.operator float();
                    float v_glsl0 = h_glsl0.operator float();

                    float error0 = abs(v_cuda0 - v_glsl0);

                    float error = error0;

                    max_error = std::max(static_cast<double>(error), max_error);
                    if (max_error > 0.004)
                        goto failure;

                    sum_error += error0;
                    count += 1;
                }
            }
        }
    }

    avg_error = sum_error / count;
    if (avg_error > 0.0008)
        goto failure;

    PrintDebugString("Test case \"%s\" passed. Max |e|: %.8f, avg |e|: %.8f\n",
                     function_name, max_error, avg_error);
    return;

failure:
    PrintDebugString("Test case \"%s\" failed. Max |e|: %.8f, avg |e|: %.8f\n",
                     function_name, max_error, avg_error);
}

void VerifyResult4(const std::vector<uint16_t>& result_cuda,
                   const std::vector<uint16_t>& result_glsl,
                   int width, int height, int depth, int n, char* function_name)
{
    assert(n == 4);

    // Compute max |e| and avg |e|.
    double max_error = 0.0;
    double sum_error = 0.0;
    double avg_error = 0.0;
    int count = 0;
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                for (int l = 0; l < n; l += n) {
                    int index =
                        i * width * height * n + j * width * n + k * n + l;

                    half h_cuda0; h_cuda0.setBits(result_cuda[index]);
                    half h_glsl0; h_glsl0.setBits(result_glsl[index]);
                    half h_cuda1; h_cuda1.setBits(result_cuda[index + 1]);
                    half h_glsl1; h_glsl1.setBits(result_glsl[index + 1]);
                    half h_cuda2; h_cuda2.setBits(result_cuda[index + 2]);
                    half h_glsl2; h_glsl2.setBits(result_glsl[index + 2]);
                    half h_cuda3; h_cuda3.setBits(result_cuda[index + 3]);
                    half h_glsl3; h_glsl3.setBits(result_glsl[index + 3]);

                    float v_cuda0 = h_cuda0.operator float();
                    float v_glsl0 = h_glsl0.operator float();
                    float v_cuda1 = h_cuda1.operator float();
                    float v_glsl1 = h_glsl1.operator float();
                    float v_cuda2 = h_cuda2.operator float();
                    float v_glsl2 = h_glsl2.operator float();

                    float error0 = abs(v_cuda0 - v_glsl0);
                    float error1 = abs(v_cuda1 - v_glsl1);
                    float error2 = abs(v_cuda2 - v_glsl2);

                    float error = std::max(error0, error1);
                    error = std::max(error, error2);

                    max_error = std::max(static_cast<double>(error), max_error);
                    if (max_error > 0.004)
                        goto failure;

                    sum_error += error0 + error1 + error2;
                    count += 3;
                }
            }
        }
    }

    avg_error = sum_error / count;
    if (avg_error > 0.0008)
        goto failure;

    PrintDebugString("Test case \"%s\" passed. Max |e|: %.8f, avg |e|: %.8f\n",
                     function_name, max_error, avg_error);
    return;

failure:
    PrintDebugString("Test case \"%s\" failed. Max |e|: %.8f, avg |e|: %.8f\n",
                     function_name, max_error, avg_error);
}

bool InitializeSimulators(FluidSimulator* sim_cuda, FluidSimulator* sim_glsl)
{
    do {
        sim_cuda->set_graphics_lib(GRAPHICS_LIB_CUDA);
        bool result = sim_cuda->Init();
        assert(result);
        if (!result)
            break;

        sim_glsl->set_graphics_lib(GRAPHICS_LIB_GLSL);
        result = sim_glsl->Init();
        assert(result);
        if (!result)
            break;

        return true;
    } while (0);

    PrintDebugString("Failed to initialize simulators.\n");
    return false;
}

void InitializeVolume4(GraphicsVolume* cuda_volume, GraphicsVolume* glsl_volume,
                       int width, int height, int depth, int n, int pitch,
                       int size)
{
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    int pos = 0;
    for (auto& i : test_data)
        i = (pos++ % n) == 3 ? 0 : half(random()).bits();

    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    CudaCore::CopyToVolume(cuda_volume->cuda_volume()->dev_array(),
                           &test_data[0], pitch, volume_size);

    glsl_volume->gl_texture()->TexImage3D(&test_data[0]);
}

void InitializeVolume1(GraphicsVolume* cuda_volume, GraphicsVolume* glsl_volume,
                       int width, int height, int depth, int n, int pitch,
                       int size)
{
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    for (auto& i : test_data)
        i = half(random()).bits();

    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    CudaCore::CopyToVolume(cuda_volume->cuda_volume()->dev_array(),
                           &test_data[0], pitch, volume_size);

    glsl_volume->gl_texture()->TexImage3D(&test_data[0]);
}

void InitializeDensityVolume(GraphicsVolume* cuda_volume,
                             GraphicsVolume* glsl_volume, int size)
{
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    for (auto& i : test_data)
        i = half(random()).bits();

    // Volumes that registered to CUDA can not be fed??
    cuda_volume->gl_texture()->TexImage3D(&test_data[0]);
    glsl_volume->gl_texture()->TexImage3D(&test_data[0]);
}

}

void FluidUnittest::TestDensityAdvection(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!InitializeSimulators(&sim_cuda, &sim_glsl))
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
    InitializeVolume4(sim_cuda.velocity_.get(), sim_glsl.velocity_.get(), width,
                      height, depth, n_v, pitch_v, size_v);
    InitializeDensityVolume(sim_cuda.density_.get(), sim_glsl.density_.get(),
                            size_d);

    sim_cuda.AdvectDensity(kTimeStep);

    // Copy the result back to CPU.
    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    std::vector<uint16_t> result_cuda(size_d, 0);
    sim_cuda.density_->gl_texture()->GetTexImage(&result_cuda[0]);

    sim_glsl.AdvectDensity(kTimeStep);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_glsl(size_d, 0);
    sim_glsl.density_->gl_texture()->GetTexImage(&result_glsl[0]);

    VerifyResult1(result_cuda, result_glsl, width, height, depth, n_d,
                  __FUNCTION__);
}

void FluidUnittest::TestTemperatureAdvection(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n_v = 4;
    int n_1 = 1;
    int pitch_v = width * sizeof(uint16_t) * n_v;
    int pitch_1 = width * sizeof(uint16_t) * n_1;
    int size_v = pitch_v * height * depth;
    int size_1 = pitch_1 * height * depth;

    // Copy the initialized data to GPU.
    InitializeVolume4(sim_cuda.velocity_.get(), sim_glsl.velocity_.get(), width,
                      height, depth, n_v, pitch_v, size_v);
    InitializeVolume1(sim_cuda.temperature_.get(), sim_glsl.temperature_.get(),
                      width, height, depth, n_1, pitch_1, size_1);

    sim_cuda.AdvectTemperature(kTimeStep);

    // Copy the result back to CPU.
    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    std::vector<uint16_t> result_cuda(size_1, 0);
    CudaCore::CopyFromVolume(&result_cuda[0], pitch_1,
                             sim_cuda.temperature_->cuda_volume()->dev_array(),
                             volume_size);

    sim_glsl.AdvectTemperature(kTimeStep);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_glsl(size_1, 0);
    sim_glsl.temperature_->gl_texture()->GetTexImage(&result_glsl[0]);

    VerifyResult1(result_cuda, result_glsl, width, height, depth, n_1,
                  __FUNCTION__);
}

void FluidUnittest::TestVelocityAdvection(int random_seed)
{
    srand(random_seed);

    FluidSimulator sim_cuda;
    FluidSimulator sim_glsl;
    if (!InitializeSimulators(&sim_cuda, &sim_glsl))
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;

    // Copy the initialized data to GPU.
    InitializeVolume4(sim_cuda.velocity_.get(), sim_glsl.velocity_.get(), width,
                      height, depth, n, pitch, size);
    sim_cuda.AdvectVelocity(kTimeStep);

    // Copy the result back to CPU.
    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    std::vector<uint16_t> result_cuda(size, 0);
    CudaCore::CopyFromVolume(&result_cuda[0], pitch,
                             sim_cuda.velocity_->cuda_volume()->dev_array(),
                             volume_size);

    sim_glsl.AdvectVelocity(kTimeStep);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_glsl(size, 0);
    sim_glsl.velocity_->gl_texture()->GetTexImage(&result_glsl[0]);

    VerifyResult4(result_cuda, result_glsl, width, height, depth, n,
                  __FUNCTION__);
}
