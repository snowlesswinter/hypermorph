#include "fluid_impl_cuda_pure.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>
#include <driver_types.h>

#include "graphics_resource.h"
#include "../vmath.hpp"

extern void LaunchAdvectVelocityPure(void* dest_array, void* velocity_array,
                                     float time_step, float dissipation,
                                     int3 volume_size);

namespace
{
int3 FromVmathVector(const vmath::Vector3& v)
{
    return make_int3(static_cast<int>(v.getX()), static_cast<int>(v.getY()),
                     static_cast<int>(v.getZ()));
}
} // Anonymous namespace.

FluidImplCudaPure::FluidImplCudaPure()
{

}

FluidImplCudaPure::~FluidImplCudaPure()
{
}

void FluidImplCudaPure::AdvectVelocity(cudaPitchedPtr* dest,
                                       cudaPitchedPtr* velocity,
                                       float time_step, float dissipation,
                                       const vmath::Vector3& volume_size)
{
    LaunchAdvectVelocityPure(dest->ptr, velocity->ptr, time_step, dissipation,
                             FromVmathVector(volume_size));
}