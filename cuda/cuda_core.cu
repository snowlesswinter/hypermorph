#include "cuda_core.h"

#include <cassert>

#include "opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "graphics_resource.h"

// cudaReadModeNormalizedFloat
// cudaReadModeElementType
texture<ushort1, cudaTextureType3D, cudaReadModeNormalizedFloat> in_tex;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_coarse;

__global__ void AbsoluteKernel(float* out_data, int w, int h, int d)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int index = block_offset * blockDim.x*blockDim.y*blockDim.z +
        blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
    float3 coord;
    coord.x = (float(blockIdx.x) * blockDim.x + threadIdx.x + 0.5f) / w;
    coord.y = (float(blockIdx.y) * blockDim.y + threadIdx.y + 0.5f) / h;
    coord.z = (float(blockIdx.z) * blockDim.z + threadIdx.x + 0.5f) / d;

    float1 cc = tex3D(in_tex, coord.x, coord.y, coord.z);
    out_data[index] = cc.x;
}

__global__ void ProlongatePackedKernel(ushort4* out_data,
                                       int num_of_block_per_slice,
                                       int tile_height, int tile_width,
                                       int3 coarse_dim, float3 inverse_size_c)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    int index = block_offset * blockDim.x*blockDim.y*blockDim.z +
        blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;

    int x = threadIdx.z * tile_width + threadIdx.x;
    int z = block_offset / num_of_block_per_slice;
    int y = (block_offset - z) * tile_height + threadIdx.y;

    int3 f_coord_int = make_int3(x, y, z);
    float3 c = make_float3(x * 0.5f, y * 0.5f, z * 0.5f);

    int odd_x = x - ((x >> 1) << 1);
    int odd_y = y - ((y >> 1) << 1);
    int odd_z = z - ((z >> 1) << 1);

    float t_x = -1.0f * (1 - odd_x) * 0.08333333f;
    float t_y = -1.0f * (1 - odd_y) * 0.08333333f;
    float t_z = -1.0f * (1 - odd_z) * 0.08333333f;

    float3 t_c = make_float3(c.x + t_x, c.y + t_y, c.z + t_z);
    float3 n_c = make_float3(t_c.x * inverse_size_c.x, t_c.y * inverse_size_c.y,
                             t_c.z * inverse_size_c.z);
    float4 f_result = tex3D(prolongate_coarse, n_c.x, n_c.y, n_c.z);
    ushort4 result = make_ushort4(__float2half_rn(f_result.x), 0, 0, 0);
    out_data[index] = result;
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

int CudaCore::RegisterGLImage(unsigned int texture, unsigned int target,
                              GraphicsResource* graphics_res)
{
    cudaError_t result = cudaGraphicsGLRegisterImage(
        graphics_res->Receive(), texture, target,
        cudaGraphicsRegisterFlagsNone);
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

void CudaCore::ProlongatePacked(GraphicsResource* coarse,
                                GraphicsResource* fine, int width)
{
    cudaGraphicsResource_t res[2] = {coarse->resource(), fine->resource()};
    cudaError_t result = cudaGraphicsMapResources(2, res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    ushort4* dest_array = nullptr;
    size_t size = 0;
    result = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dest_array), &size, res[1]);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaArray* src_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&src_array, res[0], 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    prolongate_coarse.normalized = true;
    prolongate_coarse.filterMode = cudaFilterModeLinear;
    prolongate_coarse.addressMode[0] = cudaAddressModeClamp;
    prolongate_coarse.addressMode[1] = cudaAddressModeClamp;
    prolongate_coarse.addressMode[2] = cudaAddressModeClamp;
    prolongate_coarse.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_coarse, src_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    int fine_width = width * 2;
    dim3 block(8, 8, fine_width / 8);
    dim3 grid(fine_width / block.x, fine_width / block.y, fine_width / block.z);
    int3 coarse_dim = make_int3(width, width, width);
    int num_of_block_per_slice = fine_width / 8;
    int tile_height = 8;
    int tile_width = 8;
    float3 inverse_size_c = make_float3(1.0f / width, 1.0f / width, 1.0f / width);
    ProlongatePackedKernel<<<grid, block>>>(dest_array, num_of_block_per_slice,
                                            tile_height, tile_width, coarse_dim,
                                            inverse_size_c);
    cudaGraphicsUnmapResources(2, res);
}