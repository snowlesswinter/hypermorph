#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

namespace Vectormath
{
namespace Aos
{
class Vector3;
class Vector4;
}
}

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
                                         const Vectormath::Aos::Vector3& extent,
                                         int num_of_components, int byte_width);
    static bool AllocVolumeMemory(cudaArray** result,
                                  const Vectormath::Aos::Vector3& extent,
                                  int num_of_components, int byte_width);
    static void FreeVolumeInPlaceMemory(cudaPitchedPtr* mem);
    static void FreeVolumeMemory(cudaArray* mem);

    static void ClearVolume(cudaArray* dest,
                            const Vectormath::Aos::Vector4& value,
                            const Vectormath::Aos::Vector3& volume_size);
    static void CopyFromVolume(void* dest, size_t size_in_bytes, size_t pitch,
                               cudaArray* source,
                               const Vectormath::Aos::Vector3& volume_size);
    static void CopyToVolume(cudaArray* dest, void* source,
                             size_t size_in_bytes, size_t pitch,
                             const Vectormath::Aos::Vector3& volume_size);

    int RegisterGLImage(unsigned int texture, unsigned int target,
                        GraphicsResource* graphics_res);
    int RegisterGLBuffer(unsigned int buffer, GraphicsResource* graphics_res);
    void UnregisterGLResource(GraphicsResource* graphics_res);
};

#endif // _CUDA_CORE_H_