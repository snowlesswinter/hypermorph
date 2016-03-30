#include "stdafx.h"
#include "fluid_unittest.h"

#include <cassert>

#include "fluid_simulator.h"

void FluidUnittest::Test()
{
    FluidSimulator sim_cuda;
    sim_cuda.set_graphics_lib(GRAPHICS_LIB_CUDA);
    bool result = sim_cuda.Init();
    assert(result);
    if (!result)
        return;

    sim_cuda.AdvectDensity(0.1f);


    FluidSimulator sim_glsl;
    sim_glsl.set_graphics_lib(GRAPHICS_LIB_GLSL);
    result = sim_glsl.Init();
    assert(result);
    if (!result)
        return;

    sim_glsl.AdvectDensity(0.1f);


}
