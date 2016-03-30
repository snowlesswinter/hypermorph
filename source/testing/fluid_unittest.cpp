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
float random()
{
    int l = 10000;
    double r = static_cast<double>(rand() % l - (l >> 1)) / l;
    return static_cast<float>(r) * 10.0f;
}
}

void FluidUnittest::Test()
{
    srand(0x97538642);

    FluidSimulator sim_cuda;
    sim_cuda.set_graphics_lib(GRAPHICS_LIB_CUDA);
    bool result = sim_cuda.Init();
    assert(result);
    if (!result)
        return;

    FluidSimulator sim_glsl;
    sim_glsl.set_graphics_lib(GRAPHICS_LIB_GLSL);
    result = sim_glsl.Init();
    assert(result);
    if (!result)
        return;

    int width = sim_cuda.velocity_->GetWidth();
    int height = sim_cuda.velocity_->GetHeight();
    int depth = sim_cuda.velocity_->GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    int pos = 0;
    for (auto& i : test_data) {
        i = (pos++ % n) == 3 ? 0 : half(random()).bits();
    }

    // Copy the initialized data to GPU.
    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    CudaCore::CopyToVolume(sim_cuda.velocity_->cuda_volume()->dev_array(),
                           &test_data[0], size, pitch, volume_size);
    sim_cuda.AdvectVelocity(0.33f);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_cuda(size, 0);
    CudaCore::CopyFromVolume(&result_cuda[0], size, pitch,
                             sim_cuda.velocity_->cuda_volume()->dev_array(),
                             volume_size);

    // Copy the initialized data to GPU.
    sim_glsl.velocity_->gl_texture()->TexImage3D(&test_data[0]);
    sim_glsl.AdvectVelocity(0.33f);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_glsl(size, 0);
    sim_glsl.velocity_->gl_texture()->GetTexImage(&result_glsl[0]);

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
                     __FUNCTION__, max_error, avg_error);
    return;

failure:
    PrintDebugString("Test case \"%s\" failed. Max |e|: %.8f, avg |e|: %.8f\n",
                     __FUNCTION__, max_error, avg_error);
}
