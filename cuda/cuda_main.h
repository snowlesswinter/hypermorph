#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

struct cudaGraphicsResource;
class GLTexture;
class CudaMain
{
public:
    static CudaMain* Instance();

    CudaMain();
    ~CudaMain();

    int RegisterGLImage(const GLTexture& texture);

private:
    cudaGraphicsResource* graphics_res_;
};

#endif // _CUDA_MAIN_H_