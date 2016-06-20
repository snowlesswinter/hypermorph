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

#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

#include "aux_buffer_manager.h"
#include "block_arrangement.h"
#include "third_party/glm/fwd.hpp"

struct cudaGraphicsResource;
struct cudaArray;
struct cudaPitchedPtr;
class GraphicsResource;
class CudaCore
{
public:
    CudaCore();
    ~CudaCore();

    bool Init();

    static bool AllocMemPiece(void** result, int size);
    static bool AllocVolumeInPlaceMemory(cudaPitchedPtr** result,
                                         const glm::ivec3& extent,
                                         int num_of_components, int byte_width);
    static bool AllocVolumeMemory(cudaArray** result, const glm::ivec3& extent,
                                  int num_of_components, int byte_width);
    static void FreeMemPiece(void* mem);
    static void FreeVolumeInPlaceMemory(cudaPitchedPtr* mem);
    static void FreeVolumeMemory(cudaArray* mem);

    static void CopyFromVolume(void* dest, size_t pitch, cudaArray* source,
                               const glm::ivec3& volume_size);
    static void CopyToVolume(cudaArray* dest, void* source, size_t pitch,
                             const glm::ivec3& volume_size);
    static void Raycast(GraphicsResource* dest, cudaArray* density,
                        const glm::mat4& model_view,
                        const glm::ivec2& surface_size,
                        const glm::vec3& eye_pos, const glm::vec3& light_color,
                        const glm::vec3& light_pos, float light_intensity,
                        float focal_length, int num_samples,
                        int num_light_samples, float absorption,
                        float density_factor, float occlusion_factor);

    void ClearVolume(cudaArray* dest, const glm::vec4& value,
                     const glm::ivec3& volume_size);
    int RegisterGLImage(unsigned int texture, unsigned int target,
                        GraphicsResource* graphics_res);
    int RegisterGLBuffer(unsigned int buffer, GraphicsResource* graphics_res);
    void UnregisterGLResource(GraphicsResource* graphics_res);

    void FlushProfilingData();
    void Sync();

    BlockArrangement* block_arrangement() { return &block_arrangement_; }
    AuxBufferManager* buffer_manager() { return &buffer_manager_; }

private:
    BlockArrangement block_arrangement_;
    AuxBufferManager buffer_manager_;
};

#endif // _CUDA_CORE_H_