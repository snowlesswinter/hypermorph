//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>

#include "cuda/cuda_common_host.h"
#include "cuda/graphics_resource.h"
#include "cuda/kernel_launcher.h"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec3.hpp"

namespace
{
uint3 FromGlmVector(const glm::ivec3& v)
{
    return make_uint3(static_cast<uint>(v.x), static_cast<uint>(v.y),
                      static_cast<uint>(v.z));
}
} // Anonymous namespace.

CudaCore::CudaCore()
    : block_arrangement_()
    , buffer_manager_()
    , rand_helper_()
{

}

CudaCore::~CudaCore()
{
    cudaDeviceReset();
}

bool CudaCore::Init()
{
    int dev_id = findCudaGLDevice(0, nullptr);
    block_arrangement_.Init(dev_id);

    cudaProfilerStart();

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
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
    // Believe it or not: using another CudaCore instance would cause
    //                    cudaGraphicsMapResources() crash or returning
    //                    unknown error!
    //                    This shit just tortured me for a whole day.
    //
    // So, don't treat .cu file as normal cpp files, CUDA must has done
    // something dirty with it. Just put as less as cpp code inside it
    // as possible.
    cudaError_t result = cudaGraphicsGLRegisterBuffer(
        graphics_res->Receive(), buffer, cudaGraphicsMapFlagsWriteDiscard);
    assert(result == cudaSuccess);
    return result == cudaSuccess ? 0 : -1;
}

void CudaCore::UnregisterGLResource(GraphicsResource* graphics_res)
{
    cudaError_t e = cudaGraphicsUnregisterResource(graphics_res->resource());
    assert(e == cudaSuccess);
}

bool CudaCore::AllocLinearMem(void** result, int size)
{
    return cudaMalloc(result, size) == cudaSuccess;
}

bool CudaCore::AllocMemPiece(void** result, int size)
{
    return cudaMalloc(result, size) == cudaSuccess;
}

bool CudaCore::AllocVolumeInPlaceMemory(cudaPitchedPtr** result,
                                        const glm::ivec3& extent,
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
        num_of_components * byte_width * extent.x, extent.y, extent.z);

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

bool CudaCore::AllocVolumeMemory(cudaArray** result, const glm::ivec3& extent,
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

    cudaExtent cuda_ext = make_cudaExtent(extent.x, extent.y, extent.z);
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

void CudaCore::FreeMemPiece(void* mem)
{
    if (mem) {
        cudaFree(mem);
    }
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

void CudaCore::ClearVolume(cudaArray* dest, const glm::vec4& value,
                           const glm::ivec3& volume_size)
{
    kern_launcher::ClearVolume(dest,
                               make_float4(value.x, value.y, value.z, value.w),
                               FromGlmVector(volume_size), &block_arrangement_);

}

void CudaCore::CopyFromVolume(void* dest, size_t pitch, cudaArray* source, 
                              const glm::ivec3& volume_size)
{
    cudaDeviceSynchronize();

    cudaMemcpy3DParms cpy_parms = {};
    cpy_parms.dstPtr.ptr = dest;
    cpy_parms.dstPtr.pitch = pitch;
    cpy_parms.dstPtr.xsize = volume_size.x;
    cpy_parms.dstPtr.ysize = volume_size.y;
    cpy_parms.srcArray = source;
    cpy_parms.extent.width = volume_size.x;
    cpy_parms.extent.height = volume_size.y;
    cpy_parms.extent.depth = volume_size.z;
    cpy_parms.kind = cudaMemcpyDeviceToHost;
    cudaError_t e = cudaMemcpy3D(&cpy_parms);
    assert(e == cudaSuccess);
}

void CudaCore::CopyToVolume(cudaArray* dest, void* source, size_t pitch,
                            const glm::ivec3& volume_size)
{
    cudaMemcpy3DParms cpy_parms = {};
    cpy_parms.srcPtr.ptr = source;
    cpy_parms.srcPtr.pitch = pitch;
    cpy_parms.srcPtr.xsize = volume_size.x;
    cpy_parms.srcPtr.ysize = volume_size.y;
    cpy_parms.dstArray = dest;
    cpy_parms.extent.width = volume_size.x;
    cpy_parms.extent.height = volume_size.y;
    cpy_parms.extent.depth = volume_size.z;
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

void CudaCore::Raycast(GraphicsResource* dest, cudaArray* density,
                       const glm::mat4& inv_rotation,
                       const glm::ivec2& surface_size,
                       const glm::vec3& eye_pos, const glm::vec3& light_color,
                       const glm::vec3& light_pos, float light_intensity,
                       float focal_length, const glm::vec2& screen_size,
                       int num_samples, int num_light_samples, float absorption,
                       float density_factor, float occlusion_factor,
                       const glm::vec3& volume_size)
{
    cudaGraphicsResource_t res[] = {
        dest->resource()
    };
    cudaError_t result = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                                  res);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    // Destination texture.
    cudaArray* dest_array = nullptr;
    result = cudaGraphicsSubResourceGetMappedArray(&dest_array,
                                                   dest->resource(), 0, 0);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    kern_launcher::Raycast(dest_array, density, inv_rotation, surface_size,
                           eye_pos, light_color, light_pos, light_intensity,
                           focal_length, screen_size, num_samples,
                           num_light_samples, absorption, density_factor,
                           occlusion_factor, volume_size);

    cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
}

void CudaCore::CopyVolumeAsync(cudaArray* dest, cudaArray* source,
                               const glm::ivec3& volume_size)
{
    uint3 size = make_uint3(volume_size.x, volume_size.y, volume_size.z);
    ::CopyVolumeAsync(dest, source, size);
}

void CudaCore::CopyToVbo(GraphicsResource* point_vbo,
                         GraphicsResource* extra_vbo, uint16_t* pos_x,
                         uint16_t* pos_y, uint16_t* pos_z, uint16_t* density,
                         uint16_t* temperature, float crit_density,
                         int* num_of_active_particles, int num_of_particles)
{
    cudaGraphicsResource_t res[] = {
        point_vbo->resource(),
        extra_vbo->resource(),
    };

    cudaError_t e = cudaGraphicsMapResources(sizeof(res) / sizeof(res[0]),
                                             res);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    void* p = nullptr;
    size_t bytes = 0;
    e = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&p),
                                             &bytes, res[0]);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    void* p1 = nullptr;
    e = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&p1),
                                             &bytes, res[1]);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    kern_launcher::CopyToVbo(p, p1, pos_x, pos_y, pos_z, density, temperature,
                             crit_density, num_of_active_particles,
                             num_of_particles, &block_arrangement_);
    e = cudaGraphicsUnmapResources(sizeof(res) / sizeof(res[0]), res);
    assert(e == cudaSuccess);
}
