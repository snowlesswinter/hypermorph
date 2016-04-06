#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "graphics_resource.h"
#include "../vmath.hpp"

extern void LaunchClearVolumeKernel(cudaArray* dest_array, float4 value,
                                    int3 volume_size);

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

    cudaProfilerStart();
    return 0;
}

int CudaCore::RegisterGLImage(unsigned int texture, unsigned int target,
                              GraphicsResource* graphics_res)
{
    cudaError_t result = cudaGraphicsGLRegisterImage(
        graphics_res->Receive(), texture, target,
        cudaGraphicsRegisterFlagsSurfaceLoadStore);
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

// About the CUDA texture, here is another valuable article which collaborates
// some deep insight of the mechanism:
// http://stackoverflow.com/questions/6647915/cuda-texture-memory-space
//
// Which says:
//
// "The resulting CUDA array contains a spatially ordered version
// of the linear source, stored in global memory in some sort of (undocumented)
// space filling curve.",
//
// and also njuffa gave a confirmation that filtering is also enabled in
// cudaArray bound from linear memory.
//
// One more article from the CUDA forum:
// https://devtalk.nvidia.com/default/topic/469992/why-whould-i-want-to-use-surfaces-instead-of-textures-or-global-memory-/
//
// In which the fact that surface reads are cached is confirmed, and the
// question about binding surface and texture references to the same cudaArray
// is clarified by Simon Green:
//
// "If you want texture features like interpolation and normalized coordinates
// you can always bind a texture reference to the same CUDA array.".
//
// We may use a table to conclude all of the above:
// +------------------------------+---------+-----------+-------+
// |                              | reorder | filtering | cache |
// +------------------------------+---------+-----------+-------+
// | cudaArray                    |    Y    |     Y     |   Y   |
// +------------------------------+---------+-----------+-------+
// | cudaArray(bound to surface)  |    N    |     Y     |   Y   |
// +------------------------------+---------+-----------+-------+
// | linear mem                   |    N    |     Y     |   Y   |
// +------------------------------+---------+-----------+-------+

bool CudaCore::AllocVolumeMemory(cudaArray** result,
                                 const vmath::Vector3& extent,
                                 int num_of_components, int byte_width)
{
    if (byte_width != 2 && byte_width != 4)
        return false; // Currently not supported.

    cudaChannelFormatDesc desc;
    switch (num_of_components) {
        case 1:
            desc = byte_width == 2 ?
                cudaCreateChannelDescHalf() : cudaCreateChannelDesc<float>();
            break;
        case 2:
            desc = byte_width == 2 ?
                cudaCreateChannelDescHalf2() : cudaCreateChannelDesc<float2>();
            break;
        case 4:
            desc = byte_width == 2 ?
                cudaCreateChannelDescHalf4() : cudaCreateChannelDesc<float4>();
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

void CudaCore::FreeVolumeInPlaceMemory(cudaPitchedPtr* mem)
{
    if (mem) {
        cudaFree(mem->ptr);
        delete mem;
    }
}

void CudaCore::FreeVolumeMemory(cudaArray* mem)
{
    cudaFreeArray(mem);
}

void CudaCore::ClearVolume(cudaArray* dest, const vmath::Vector4& value,
                           const vmath::Vector3& volume_size)
{
    LaunchClearVolumeKernel(
        dest,
        make_float4(value.getX(), value.getY(), value.getZ(), value.getW()),
        FromVmathVector(volume_size));

}

void CudaCore::CopyFromVolume(void* dest, size_t pitch, cudaArray* source, 
                              const vmath::Vector3& volume_size)
{
    cudaDeviceSynchronize();

    cudaMemcpy3DParms cpy_parms = {};
    cpy_parms.dstPtr.ptr = dest;
    cpy_parms.dstPtr.pitch = pitch;
    cpy_parms.dstPtr.xsize = static_cast<size_t>(volume_size.getX());
    cpy_parms.dstPtr.ysize = static_cast<size_t>(volume_size.getY());
    cpy_parms.srcArray = source;
    cpy_parms.extent.width = static_cast<size_t>(volume_size.getX());
    cpy_parms.extent.height = static_cast<size_t>(volume_size.getY());
    cpy_parms.extent.depth = static_cast<size_t>(volume_size.getX());
    cpy_parms.kind = cudaMemcpyDeviceToHost;
    cudaError_t e = cudaMemcpy3D(&cpy_parms);
    assert(e == cudaSuccess);
}

void CudaCore::CopyToVolume(cudaArray* dest, void* source, size_t pitch,
                            const vmath::Vector3& volume_size)
{
    cudaMemcpy3DParms cpy_parms = {};
    cpy_parms.srcPtr.ptr = source;
    cpy_parms.srcPtr.pitch = pitch;
    cpy_parms.srcPtr.xsize = static_cast<size_t>(volume_size.getX());
    cpy_parms.srcPtr.ysize = static_cast<size_t>(volume_size.getY());
    cpy_parms.dstArray = dest;
    cpy_parms.extent.width = static_cast<size_t>(volume_size.getX());
    cpy_parms.extent.height = static_cast<size_t>(volume_size.getY());
    cpy_parms.extent.depth = static_cast<size_t>(volume_size.getX());
    cpy_parms.kind = cudaMemcpyHostToDevice;
    cudaError_t e = cudaMemcpy3D(&cpy_parms);
    assert(e == cudaSuccess);
}

void CudaCore::FlushProfilingData()
{
    // Still not able to make visual profiler work. Maybe it's a problem with
    // the graphics drivers:
    // https://devtalk.nvidia.com/default/topic/630905/nsight-visual-studio-edition/-quot-profile-cuda-application-quot-always-fails-with-quot-no-kernel-launches-captured-quot-/

    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void CudaCore::Sync()
{
    cudaDeviceSynchronize();
}