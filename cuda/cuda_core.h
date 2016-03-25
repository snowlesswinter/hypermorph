#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}

struct cudaGraphicsResource;
struct cudaPitchedPtr;
class GraphicsResource;
class CudaCore
{
public:
    CudaCore();
    ~CudaCore();

    bool Init();
    static bool AllocVolumeMemory(cudaPitchedPtr** result,
                                  const Vectormath::Aos::Vector3& extent,
                                  int num_of_components, int byte_width);
    static void FreeVolumeMemory(cudaPitchedPtr* mem);

    int RegisterGLImage(unsigned int texture, unsigned int target,
                        GraphicsResource* graphics_res);
    int RegisterGLBuffer(unsigned int buffer, GraphicsResource* graphics_res);
    void UnregisterGLResource(GraphicsResource* graphics_res);
    void Absolute(GraphicsResource* graphics_res, unsigned int aa);
    void ProlongatePacked(GraphicsResource* coarse, GraphicsResource* fine,
                          GraphicsResource* out_pbo,
                          const Vectormath::Aos::Vector3& volume_size_fine);
};

#endif // _CUDA_CORE_H_