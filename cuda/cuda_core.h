#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

struct cudaGraphicsResource;
class GLTexture;
class GraphicsResource;
class CudaCore
{
public:
    CudaCore();
    ~CudaCore();

    bool Init();
    int RegisterGLImage(const GLTexture& texture,
                        GraphicsResource* graphics_res);
    void UnregisterGLImage(GraphicsResource* graphics_res);
    void Absolute();

private:
};

#endif // _CUDA_CORE_H_