#include "fluid_impl_cuda.h"

#include <cassert>

#include "opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>

#include "graphics_resource.h"
#include "../vmath.hpp"

extern void LaunchAdvectVelocity(ushort4* dest_array, cudaArray* velocity_array,
                                 float time_step, float dissipation,
                                 int3 volume_size);
extern void LaunchAdvect(ushort* dest_array, cudaArray* velocity_array,
                         cudaArray* source_array, float time_step,
                         float dissipation, int3 volume_size);
extern void LaunchApplyBuoyancy(ushort4* dest_array, cudaArray* velocity_array,
                                cudaArray* temperature_array, float time_step,
                                float ambient_temperature, float accel_factor,
                                float gravity, int3 volume_size);
extern void LaunchApplyImpulse(ushort* dest_array, cudaArray* original_array,
                               float3 center_point, float3 hotspot,
                               float radius, float value, int3 volume_size);
extern void LaunchComputeDivergence(ushort4* dest_array,
                                    cudaArray* velocity_array,
                                    float half_inverse_cell_size,
                                    int3 volume_size);
extern void LaunchSubstractGradient(ushort4* dest_array,
                                    cudaArray* velocity_array,
                                    cudaArray* packed_array,
                                    float gradient_scale, int3 volume_size);
extern void LaunchDampedJacobi(ushort4* dest_array, cudaArray* packed_array,
                               float one_minus_omega,
                               float minus_square_cell_size,
                               float omega_over_beta, int3 volume_size);
extern void LaunchRoundPassed(int* dest_array, int round, int x);

namespace
{
int3 FromVmathVector(const vmath::Vector3& v)
{
    return make_int3(static_cast<int>(v.getX()), static_cast<int>(v.getY()),
                     static_cast<int>(v.getZ()));
}
} // Anonymous namespace.

FluidImplCuda::FluidImplCuda()
{

}

FluidImplCuda::~FluidImplCuda()
{
}

void FluidImplCuda::AdvectVelocity(GraphicsResource* velocity,
                                   GraphicsResource* out_pbo, float time_step,
                                   float dissipation,
                                   const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Velocity texture.
    cudaArray* velocity_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&velocity_array,
                                                   velocity->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchAdvectVelocity(dest_array, velocity_array, time_step, dissipation,
                         FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::Advect(GraphicsResource* velocity, GraphicsResource* source,
                           GraphicsResource* out_pbo, float time_step,
                           float dissipation, const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), source->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Velocity texture.
    cudaArray* velocity_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&velocity_array,
                                                   velocity->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Source texture.
    cudaArray* source_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&source_array,
                                                   source->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchAdvect(dest_array, velocity_array, source_array, time_step,
                 dissipation, FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::ApplyBuoyancy(GraphicsResource* velocity,
                                  GraphicsResource* temperature,
                                  GraphicsResource* out_pbo, float time_step,
                                  float ambient_temperature, float accel_factor,
                                  float gravity,
                                  const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), temperature->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Velocity texture.
    cudaArray* velocity_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&velocity_array,
                                                   velocity->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Temperature texture.
    cudaArray* temperature_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&temperature_array,
                                                   temperature->resource(), 0,
                                                   0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchApplyBuoyancy(dest_array, velocity_array, temperature_array,
                        time_step, ambient_temperature, accel_factor, gravity,
                        FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::ApplyImpulse(GraphicsResource* source,
                                 GraphicsResource* out_pbo,
                                 const vmath::Vector3& center_point,
                                 const vmath::Vector3& hotspot, float radius,
                                 float value, const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        source->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort* dest_array = nullptr;                         
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Source texture.
    cudaArray* original_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&original_array,
                                                   source->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchApplyImpulse(
        dest_array, original_array,
        make_float3(center_point.getX(), center_point.getY(),
                    center_point.getZ()),
        make_float3(hotspot.getX(), hotspot.getY(), hotspot.getZ()), radius,
        value, FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::ComputeDivergence(GraphicsResource* velocity,
                                      GraphicsResource* out_pbo,
                                      float half_inverse_cell_size,
                                      const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Velocity texture.
    cudaArray* velocity_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&velocity_array,
                                                   velocity->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchComputeDivergence(
        dest_array, velocity_array, half_inverse_cell_size,
        FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::SubstractGradient(GraphicsResource* velocity,
                                      GraphicsResource* packed,
                                      GraphicsResource* out_pbo,
                                      float gradient_scale,
                                      const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), packed->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Velocity texture.
    cudaArray* velocity_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&velocity_array,
                                                   velocity->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Packed texture.
    cudaArray* packed_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&packed_array,
                                                   packed->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchSubstractGradient(
        dest_array, velocity_array, packed_array, gradient_scale,
        FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::DampedJacobi(GraphicsResource* packed,
                                 GraphicsResource* out_pbo,
                                 float one_minus_omega,
                                 float minus_square_cell_size,
                                 float omega_over_beta,
                                 const vmath::Vector3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        packed->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    ushort4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Velocity texture.
    cudaArray* packed_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&packed_array,
                                                   packed->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchDampedJacobi(dest_array, packed_array, one_minus_omega,
                       minus_square_cell_size, omega_over_beta,
                       FromVmathVector(volume_size));

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void FluidImplCuda::RoundPassed(int round)
{
    //     if (round != 10)
    //         return;

    int* dest_array = nullptr;
    cudaError_t result = cudaMalloc(&dest_array, 4);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchRoundPassed(dest_array, round, 3);

    cudaFree(dest_array);
}
