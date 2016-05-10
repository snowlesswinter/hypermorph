#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

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

    static bool AllocVolumeInPlaceMemory(cudaPitchedPtr** result,
                                         const glm::ivec3& extent,
                                         int num_of_components, int byte_width);
    static bool AllocVolumeMemory(cudaArray** result, const glm::ivec3& extent,
                                  int num_of_components, int byte_width);
    static void FreeVolumeInPlaceMemory(cudaPitchedPtr* mem);
    static void FreeVolumeMemory(cudaArray* mem);

    static void ClearVolume(cudaArray* dest, const glm::vec4& value,
                            const glm::ivec3& volume_size);
    static void CopyFromVolume(void* dest, size_t pitch, cudaArray* source,
                               const glm::ivec3& volume_size);
    static void CopyToVolume(cudaArray* dest, void* source, size_t pitch,
                             const glm::ivec3& volume_size);
    static void Raycast(GraphicsResource* dest, cudaArray* density,
                        const glm::mat4& model_view,
                        const glm::ivec2& surface_size,
                        const glm::vec3& eye_pos, float focal_length);

    int RegisterGLImage(unsigned int texture, unsigned int target,
                        GraphicsResource* graphics_res);
    int RegisterGLBuffer(unsigned int buffer, GraphicsResource* graphics_res);
    void UnregisterGLResource(GraphicsResource* graphics_res);

    void FlushProfilingData();
    void Sync();

    BlockArrangement* block_arrangement() { return &block_arrangement_; }

private:
    BlockArrangement block_arrangement_;
};

#endif // _CUDA_CORE_H_