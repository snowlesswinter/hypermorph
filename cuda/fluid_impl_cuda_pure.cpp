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

extern void LaunchAdvectVelocityPure(cudaArray_t dest_array,
                                     cudaArray_t velocity_array,
                                     float time_step, float dissipation,
                                     int3 volume_size);
extern void LaunchAdvectPure(cudaArray_t dest_array, cudaArray_t velocity_array,
                             cudaArray_t source_array, float time_step,
                             float dissipation, int3 volume_size);

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

void FluidImplCudaPure::AdvectVelocity(cudaArray* dest, cudaArray* velocity,
                                       float time_step, float dissipation,
                                       const vmath::Vector3& volume_size)
{
    LaunchAdvectVelocityPure(dest, velocity, time_step, dissipation,
                             FromVmathVector(volume_size));
}

void FluidImplCudaPure::Advect(cudaArray* dest, cudaArray* velocity,
                               cudaArray* source, float time_step,
                               float dissipation,
                               const Vectormath::Aos::Vector3& volume_size)
{
    LaunchAdvectPure(dest, velocity, source, time_step, dissipation,
                     FromVmathVector(volume_size));
}
