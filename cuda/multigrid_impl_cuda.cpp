#include "multigrid_impl_cuda.h"

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

extern void LaunchAdvect(cudaArray_t dest_array, cudaArray_t velocity_array,
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

MultigridImplCuda::MultigridImplCuda()
{

}

MultigridImplCuda::~MultigridImplCuda()
{
}

void MultigridImplCuda::Advect(cudaArray* dest, cudaArray* velocity,
                               cudaArray* source, float time_step,
                               float dissipation,
                               const Vectormath::Aos::Vector3& volume_size)
{
    LaunchAdvect(dest, velocity, source, time_step, dissipation,
                     FromVmathVector(volume_size));
}
