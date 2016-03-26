#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "graphics_resource.h"
#include "../vmath.hpp"

extern void LaunchProlongatePacked(float4* dest_array, cudaArray* coarse_array,
                                   cudaArray* fine_array,
                                   int3 volume_size_fine);

namespace
{
int3 FromVmathVector(const vmath::Vector3& v)
{
    return make_int3(static_cast<int>(v.getX()), static_cast<int>(v.getY()),
                     static_cast<int>(v.getZ()));
}
} // Anonymous namespace.

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

void CudaCore::UnregisterGLResource(GraphicsResource* graphics_res)
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
                                GraphicsResource* out_pbo,
                                const vmath::Vector3& volume_size_fine)
{
    cudaGraphicsResource_t res[] = {
        coarse->resource(), fine->resource(), out_pbo->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
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

    LaunchProlongatePacked(dest_array, coarse_array, fine_array,
                           FromVmathVector(volume_size_fine));

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

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

bool CudaCore::AllocVolumeInPlaceMemory(cudaPitchedPtr** result,
                                        const vmath::Vector3& extent,
                                        int num_of_components, int byte_width)
{
    // I vacillated quite a while deciding which kind of device memory is my
    // best choice. Since I have a relatively rare scene using the memory that
    // the answer can not be found neither on google nor the nvidia CUDA forum.
    //
    // Firstly, I really need the optimized memory access of cudaArray that
    // speeds up the fetch to a texture(3d locality), which suggests that I
    // should not pick up cudaSurface, which access linear memory only.
    // However, I also require the ability to modify the texture in-place(that
    // is not applicable on cudaArray), otherwise I have to use a ping-pong
    // pattern to switch the read/write buffers back and forth, which doubles
    // my memory usage, that is unacceptable in this memory-dense
    // application(a dedicated temporary buffer is not an option because I
    // have to do the multigrid calculation, in which buffers are of different
    // sizes).
    //
    // This discrepancy seems to be unsolvable. And I start wondering: how
    // GLSL tackle this situation then?
    //
    // Meanwhile, I found some substantial articles on the web:
    // https://devtalk.nvidia.com/default/topic/419361/cudabindtexture2d-vs-cudabindtexturetoarray/
    // http://www.math.ntu.edu.tw/~wwang/mtxcomp2010/download/cuda_04_ykhung.pdf
    // https://devtalk.nvidia.com/default/topic/470009/3d-texture-and-memory-writes-write-memory-bound-to-3d-texture/
    // https://devtalk.nvidia.com/default/topic/368495/cuda-vs-opengl-textures-read-only-vs-read-write/
    //
    // And finally it seems to make sense to me that why the cudaArray is
    // read-only and can not be transformed to a normal memory pointer. The
    // reason is that the cudaArray has done some reordering on the memory
    // arrangement so as to increase the access locality. When it is bound to
    // a surface, this characteristics has gone, as well as the high accessing
    // speed.
    //
    // But this a just an answer, not a solution to me. And I don't believe
    // GLSL has any magic on this, but using linear memory instead. The best
    // choice for me maybe a compromise, a hybrid mode: using cudaArray for
    // most of the volumes, and cudaSurface for multigrid coarse volumes. We
    // will lose some speed in calculation, but since these volumes are
    // relatively small, it is still acceptable(at least the performance should
    // get even as GLSL, theoretically).

    cudaExtent cuda_ext = make_cudaExtent(
        num_of_components * byte_width * static_cast<int>(extent.getX()),
        static_cast<int>(extent.getY()), static_cast<int>(extent.getZ()));

    cudaPitchedPtr mem;
    cudaError_t e = cudaMalloc3D(&mem, cuda_ext);
    if (e == cudaSuccess) {
        cudaPitchedPtr* p = new cudaPitchedPtr();
        *p = mem;
        *result = p;
        return true;
    }

    return false;
}

bool CudaCore::AllocVolumeMemory(cudaArray** result,
                                 const vmath::Vector3& extent,
                                 int num_of_components, int byte_width)
{
    if (byte_width != 2)
        return false; // Currently not supported.

    cudaChannelFormatDesc desc;
    switch (num_of_components) {
        case 1:
            desc = cudaCreateChannelDescHalf();
            break;
        case 2:
            desc = cudaCreateChannelDescHalf2();
            break;
        case 4:
            desc = cudaCreateChannelDescHalf4();
            break;
        default:
            return false;
    }

    cudaExtent cuda_ext = make_cudaExtent(static_cast<int>(extent.getX()),
                                          static_cast<int>(extent.getY()),
                                          static_cast<int>(extent.getZ()));

    cudaArray* mem;

    // Maybe there is no magic at all: the memory reordering will have gone
    // with the flag set to cudaArraySurfaceLoadStore. Then what's the point
    // of cudaSurface after all?
    //
    // Fortunately, I got a confirm that surface will not be slow in writing:
    // https://devtalk.nvidia.com/default/topic/908776/cuda-programming-and-performance/writing-performance-of-surface-in-cuda/
    //
    // At least, I don't have to worry about rolling back to the global memory.

    cudaError_t e = cudaMalloc3DArray(&mem, &desc, cuda_ext,
                                      cudaArraySurfaceLoadStore);
    if (e == cudaSuccess) {
        *result = mem;
        return true;
    }

    return false;
}

void CudaCore::FreeVolumeMemory(cudaArray* mem)
{
    cudaFreeArray(mem);
}

void CudaCore::FreeVolumeInPlaceMemory(cudaPitchedPtr* mem)
{
    if (mem) {
        cudaFree(mem->ptr);
        delete mem;
    }
}