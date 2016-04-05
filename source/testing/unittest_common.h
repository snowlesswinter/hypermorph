#ifndef _UNITTEST_COMMON_H_
#define _UNITTEST_COMMON_H_

#include <utility>
#include <vector>

#include <stdint.h>

class FluidSimulator;
class GraphicsVolume;
class UnittestCommon
{
public:
    static void CollectAndVerifyResult(int width, int height, int depth,
                                       int size, int pitch, int n,
                                       uint32_t channel_mask,
                                       GraphicsVolume* cuda_volume,
                                       GraphicsVolume* glsl_volume,
                                       char* function_name);
    static bool InitializeSimulators(FluidSimulator* sim_cuda,
                                     FluidSimulator* sim_glsl);
    static void InitializeVolume1(GraphicsVolume* cuda_volume,
                                  GraphicsVolume* glsl_volume, int width,
                                  int height, int depth, int n, int pitch,
                                  int size,
                                  const std::pair<float, float>& scope);
    static void InitializeVolume4(GraphicsVolume* cuda_volume,
                                  GraphicsVolume* glsl_volume, int width,
                                  int height, int depth, int n, int pitch,
                                  int size,
                                  const std::pair<float, float>& scope);
    static float RandomFloat(const std::pair<float, float>& scope);
    static void VerifyResult1(const std::vector<uint16_t>& result_cuda,
                              const std::vector<uint16_t>& result_glsl,
                              int width, int height, int depth, int n,
                              char* function_name);

private:
    UnittestCommon();
    ~UnittestCommon();
};

#endif // _UNITTEST_COMMON_H_