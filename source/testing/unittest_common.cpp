#include "stdafx.h"
#include "unittest_common.h"

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
// In both velocity and temperature test cases, I encountered the same maximum
// difference number 0.00390625, which had drawn my attention. So I stopped
// writing more test cases and got a look into it.
// It seemed to be connected with CUDA's interpolation implementation that
// using 9-bit floating point numbers as the coefficients:
// https://devtalk.nvidia.com/default/topic/528016/cuda-programming-and-performance/accuracy-of-1d-linear-interpolation-by-cuda-texture-interpolation
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering
//
// However, if I change the scope of the random data, say, from 10.0f to 20.0f,
// the difference between CUDA and GLSL results will become much bigger,
// indicating that this is not a system error cause by either of the
// implementations.
//
// I then wrote my own interpolation for comparison. But the result is
// frustrating: the difference is even bigger than both of the previous results.
// I checked many times and didn't find anything wrong with the tri-linear
// interpolation algorithm(after all differing form those results doesn't mean
// the algorithm is wrong), and I just turned to another way: I began to
// suspect the 16-bit floating point conversion.
//
// It is possible that GLSL shaders use a different standard of half-precision,
// which means using openEXR(specified by CUDA) to convert single-precision
// floating point numbers could be a problem. So I changed the texture back to
// 32-bit floating point format, and finally I got the same results came from
// CUDA and GLSL.
//
// Piece of test results(with random seed 0x56784321, and scope [-5, 5]):
//
//  Point(0, 0, 0)
//  ----------------------------------------------------------------------------
//  cuda interpolation :
//  new_velocity{x = -2.0389745, y = 0.65754491, z = -0.085241824, w = 0}
//
//  glsl interpolation :
//  new_velocity{x = -2.03710938, y = 0.657226563, z = -0.0852050781, w = 0}
//
//  single-precision float interpolation :
//  new_velocity{x = -2.03835416, y = 0.657923460, z = -0.0852115080, w = 0}
//
//  single-precision float DIY interpolation :
//  new_velocity{x = -2.03192258, y = 0.654677153, z = -0.103225358, w = 0}
//
//  Point(36, 36, 36)
//  ----------------------------------------------------------------------------
//  cuda interpolation :
//  new_velocity{x = -3.10351563, y = -1.29296875, z = 1.08203125, w = 0}
//
//  glsl interpolation :
//  new_velocity{x = -3.10351563, y = -1.29199219, z = 1.08105469, w = 0}
//
//  single-precision float interpolation :
//  new_velocity{x = -3.10525894, y = -1.29387271, z = 1.08214724, w = 0}
//
//  single-precision float DIY interpolation :
//  new_velocity{x = -3.12161803, y = -1.29015040, z = 1.08285618, w = 0}
//
// There could be difference if the scope goes big(e.g. [-15, 15]).
//
// Conclusion:
//
// Maybe the GLSL uses a different implementation of half-floats(it's announced
// that conformed to IEEE-754 though), it is more likely that GLSL has bigger
// errors dealing with half-floats.

const float kErrorThreshold = 0.08f;

float random(const std::pair<float, float>& scope)
{
    int l = 10000;
    double r = static_cast<double>(rand() % l) / l;
    return scope.first + static_cast<float>(r)* (scope.second - scope.first);
}

void VerifyResult4(const std::vector<uint16_t>& result_cuda,
                   const std::vector<uint16_t>& result_glsl,
                   int width, int height, int depth, int n,
                   uint32_t channel_mask, char* function_name)
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
                    if (max_error > kErrorThreshold)
                        goto failure;

                    float errors[] = {error0, error1, error2};
                    int t = 0;
                    for (float e : errors) {
                        if (channel_mask & (1 << (t++))) {
                            sum_error += e;
                            count++;
                        }
                    }
                }
            }
        }
    }

    avg_error = count ? sum_error / count : 0.0;
    if (avg_error > 0.0008)
        goto failure;

    PrintDebugString("Test case \"%s\" passed. Max |e|: %.8f, avg |e|: %.8f\n",
                     function_name, max_error, avg_error);
    return;

failure:
    PrintDebugString("Test case \"%s\" failed. Max |e|: %.8f, avg |e|: %.8f\n",
                     function_name, max_error, avg_error);
}

} // Anonymous namespace.

void UnittestCommon::CollectAndVerifyResult(int width, int height, int depth,
                                            int size, int pitch, int n,
                                            uint32_t channel_mask,
                                            GraphicsVolume* cuda_volume,
                                            GraphicsVolume* glsl_volume,
                                            char* function_name)
{
    // Copy the result back to CPU.
    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    std::vector<uint16_t> result_cuda(size, 0);
    CudaCore::CopyFromVolume(&result_cuda[0], pitch,
                             cuda_volume->cuda_volume()->dev_array(),
                             volume_size);

    // Copy the result back to CPU.
    std::vector<uint16_t> result_glsl(size, 0);
    glsl_volume->gl_texture()->GetTexImage(&result_glsl[0]);

    assert(n == 1 || n == 4);
    if (n == 1)
        VerifyResult1(result_cuda, result_glsl, width, height, depth, n,
        function_name);
    else if (n == 4)
        VerifyResult4(result_cuda, result_glsl, width, height, depth, n,
        channel_mask, function_name);
}

bool UnittestCommon::InitializeSimulators(FluidSimulator* sim_cuda,
                                          FluidSimulator* sim_glsl)
{
    do
    {
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

void UnittestCommon::InitializeVolume1(GraphicsVolume* cuda_volume,
                                       GraphicsVolume* glsl_volume,
                                       int width, int height, int depth, int n,
                                       int pitch, int size,
                                       const std::pair<float, float>& scope)
{
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    for (auto& i : test_data)
        i = half(random(scope)).bits();

    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    CudaCore::CopyToVolume(cuda_volume->cuda_volume()->dev_array(),
                           &test_data[0], pitch, volume_size);

    glsl_volume->gl_texture()->TexImage3D(&test_data[0]);
}

void UnittestCommon::InitializeVolume4(GraphicsVolume* cuda_volume,
                                       GraphicsVolume* glsl_volume,
                                       int width, int height, int depth, int n,
                                       int pitch, int size,
                                       const std::pair<float, float>& scope)
{
    std::vector<uint16_t> test_data(size / sizeof(uint16_t), 0);
    int pos = 0;
    for (auto& i : test_data)
        i = (pos++ % n) == 3 ? 0 : half(random(scope)).bits();

    vmath::Vector3 volume_size(static_cast<float>(width),
                               static_cast<float>(height),
                               static_cast<float>(depth));
    CudaCore::CopyToVolume(cuda_volume->cuda_volume()->dev_array(),
                           &test_data[0], pitch, volume_size);

    glsl_volume->gl_texture()->TexImage3D(&test_data[0]);
}

float UnittestCommon::RandomFloat(const std::pair<float, float>& scope)
{
    return random(scope);
}

void UnittestCommon::VerifyResult1(const std::vector<uint16_t>& result_cuda,
                                   const std::vector<uint16_t>& result_glsl,
                                   int width, int height, int depth, int n,
                                   char* function_name)
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
                    if (max_error > kErrorThreshold)
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
