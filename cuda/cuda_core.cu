#include "cuda_core.h"

#include <cassert>

#include "glew.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "opengl/gl_texture.h"
#include "graphics_resource.h"

// cudaReadModeNormalizedFloat
// cudaReadModeElementType
texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat> in_tex;

__global__ void AbsoluteKernel(float* out_data, int w, int h, int d)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int index = block_offset * blockDim.x*blockDim.y*blockDim.z +
        blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
    float3 coord;
    coord.x = (float(blockIdx.x) * blockDim.x + threadIdx.x + 0.5f) / w;
    coord.y = (float(blockIdx.y) * blockDim.y + threadIdx.y + 0.5f) / h;
    coord.z = (float(blockIdx.z) * blockDim.z + threadIdx.x + 0.5f) / d;

    float cc = tex3D(in_tex, coord.x, coord.y, coord.z);
    out_data[index] = cc;
}

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

int CudaCore::RegisterGLImage(const GLTexture& texture,
                              GraphicsResource* graphics_res)
{
    cudaError_t result = cudaGraphicsGLRegisterImage(
        graphics_res->Receive(), texture.handle(), texture.target(),
        cudaGraphicsRegisterFlagsNone);
    assert(result == cudaSuccess);
    return result == cudaSuccess ? 0 : -1;
}

void CudaCore::UnregisterGLImage(GraphicsResource* graphics_res)
{
    cudaGraphicsUnregisterResource(graphics_res->resource());
}

void CudaCore::Absolute(GraphicsResource* graphics_res, unsigned int aa)
{
    assert(graphics_res);
    if (!graphics_res)
        return;

    float* out_data = nullptr;
    size_t out_size = 0;
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