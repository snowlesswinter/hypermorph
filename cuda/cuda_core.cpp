#include "cuda_core.h"

#include <cassert>

#include "opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "graphics_resource.h"

extern void LaunchProlongatePacked(float4* dest_array, cudaArray* coarse_array,
                                   cudaArray* fine_array, int coarse_width);
extern void LaunchAdvectVelocity(float4* dest_array, cudaArray* velocity_array,
                                 float time_step, float dissipation, int width);
extern void LaunchAdvect(float* dest_array, cudaArray* velocity_array,
                         cudaArray* source_array, float time_step,
                         float dissipation, int width);
extern void LaunchApplyBuoyancy(float4* dest_array, cudaArray* velocity_array,
                                cudaArray* temperature_array, float time_step,
                                float ambient_temperature, float accel_factor,
                                float gravity, int width);
extern void LaunchRoundPassed(int* dest_array, int round, int x);

CudaCore::CudaCore()
{

}

CudaCore::~CudaCore()
{
    cudaDeviceReset();
}

bool CudaCore::Init()
{
    int dev_id = findCudaGLDevice(0, nullptr);
    cudaDeviceProp prop = {0};
    cudaGetDeviceProperties(&prop, dev_id);
    return 0;
}

int CudaCore::RegisterGLImage(unsigned int texture, unsigned int target,
                              GraphicsResource* graphics_res)
{
    cudaError_t result = cudaGraphicsGLRegisterImage(
        graphics_res->Receive(), texture, target,
        cudaGraphicsRegisterFlagsReadOnly);
    assert(result == cudaSuccess);
    return result == cudaSuccess ? 0 : -1;
}

int CudaCore::RegisterGLBuffer(unsigned int buffer,
                               GraphicsResource* graphics_res)
{
    cudaError_t result = cudaGraphicsGLRegisterBuffer(
        graphics_res->Receive(), buffer, cudaGraphicsRegisterFlagsNone);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    return result == cudaSuccess ? 0 : -1;
}

void CudaCore::UnregisterGLImage(GraphicsResource* graphics_res)
{
    cudaGraphicsUnregisterResource(graphics_res->resource());
}

void CudaCore::Absolute(GraphicsResource* graphics_res, unsigned int aa)
{
}

#if 0
void CudaCore::Absolute(GraphicsResource* graphics_res, unsigned int aa)
{
    assert(graphics_res);
    if (!graphics_res)
        return;

    float* out_data = nullptr;
    cudaError_t result1 = cudaMalloc((void**)&out_data, 128 * 128 * 128 * 4);
    assert(result1 == cudaSuccess);
    if (result1 != cudaSuccess)
        return;
    //cudaGraphicsResource_t res1;
    //cudaError_t result1 = cudaGraphicsGLRegisterBuffer(
    //    &res1, aa, cudaGraphicsRegisterFlagsNone);
    //
    //result1 = cudaGraphicsMapResources(1, &res1);
    //assert(result1 == cudaSuccess);
    //if (result1 != cudaSuccess)
    //    return;

    //result1 = cudaGraphicsResourceGetMappedPointer(
    //    reinterpret_cast<void**>(&out_data), &out_size, res1);
    //assert(result1 == cudaSuccess);
    //if (result1 != cudaSuccess)
    //    return;

    cudaGraphicsResource_t res = graphics_res->resource();
    cudaError_t result = cudaGraphicsMapResources(1, &res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaArray* dest_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&dest_array, res, 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf();
    in_tex.normalized = true;
    in_tex.filterMode = cudaFilterModeLinear;
    in_tex.addressMode[0] = cudaAddressModeClamp;
    in_tex.addressMode[1] = cudaAddressModeClamp;
    in_tex.addressMode[2] = cudaAddressModeClamp;
    in_tex.channelDesc = desc;
    
    result = cudaBindTextureToArray(&in_tex, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(16, 16, 16);
    AbsoluteKernel<<<grid, block>>>(out_data, 128, 128, 128);

    result = cudaGetLastError();
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    float* a = new float[128 * 128 * 128];
    result = cudaMemcpy(a, out_data, 128 * 128 * 128 * 4, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    double p = 0;
    double sum = 0;
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            for (int k = 0; k < 128; k++)
            {
                p = a[i * 128 * 128 + j * 128 + k];
                sum += p;
            }
        }
    }

    cudaUnbindTexture(&in_tex);
    cudaGraphicsUnmapResources(1, &res);
}
#endif
void CudaCore::ProlongatePacked(GraphicsResource* coarse,
                                GraphicsResource* fine,
                                GraphicsResource* out_pbo, int width)
{
    cudaGraphicsResource_t res[3] = {
        coarse->resource(), fine->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(3, res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    float4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, out_pbo->resource());
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Coarse texture.
    cudaArray* coarse_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&coarse_array,
                                                   coarse->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Fine texture.
    cudaArray* fine_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&fine_array,
                                                   fine->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchProlongatePacked(dest_array, coarse_array, fine_array, width);

//     float* a = new float[128 * 128 * 128 * 4];
//     result = cudaMemcpy(a, dest_array, 128 * 128 * 128 * 4 * 4,
//                         cudaMemcpyDeviceToHost);
//     assert(result == cudaSuccess);
//     if (result != cudaSuccess)
//         return;
//
//     double p = 0;
//     double sum = 0;
//     for (int i = 0; i < 128; i++) {
//         for (int j = 0; j < 128; j++) {
//             for (int k = 0; k < 128; k++) {
//                 for (int n = 0; n < 4; n += 4) {
//                     float* z = &a[i * 128 * 128 * 4 + j * 128 * 4 + k * 4 + n];
//                     p = *z;
//                     float z0 = *z;
//                     float z1 = *(z + 1);
//                     float z2 = *(z + 2);
//                     if (z0 != k || z1 != j || z2 != i)
//                         sum += p;
//                 }
//             }
//         }
//     }
// 
//     delete[] a;

    cudaGraphicsUnmapResources(3, res);
}

void CudaCore::AdvectVelocity(GraphicsResource* velocity,
                              GraphicsResource* out_pbo, float time_step,
                              float dissipation, int width)
{
    cudaGraphicsResource_t res[2] = {
        velocity->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(2, res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    float4* dest_array = nullptr;
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
                         width);

    cudaGraphicsUnmapResources(2, res);
}

void CudaCore::Advect(GraphicsResource* velocity, GraphicsResource* source,
                      GraphicsResource* out_pbo, float time_step,
                      float dissipation, int width)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), source->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(3, res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    float* dest_array = nullptr;
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
                 dissipation, width);

    cudaGraphicsUnmapResources(3, res);
}

void CudaCore::ApplyBuoyancy(GraphicsResource* velocity,
                             GraphicsResource* temperature,
                             GraphicsResource* out_pbo, float time_step,
                             float ambient_temperature, float accel_factor,
                             float gravity, int width)
{
    cudaGraphicsResource_t res[] = {
        velocity->resource(), temperature->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(3, res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Output to pbo.
    float4* dest_array = nullptr;
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
    cudaArray* temperature_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&temperature_array,
                                                   temperature->resource(), 0,
                                                   0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    LaunchApplyBuoyancy(dest_array, velocity_array, temperature_array,
                        time_step, ambient_temperature, accel_factor, gravity,
                        width);

    cudaGraphicsUnmapResources(3, res);
}

void CudaCore::RoundPassed(int round)
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
