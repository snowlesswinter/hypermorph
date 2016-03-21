#ifndef _CUDA_MAIN_H_
#define _CUDA_MAIN_H_

#include <map>
#include <memory>

class CudaCore;
class GLTexture;
class GraphicsResource;
class CudaMain
{
public:
    static CudaMain* Instance();

    CudaMain();
    ~CudaMain();

    bool Init();
    int RegisterGLImage(const std::shared_ptr<GLTexture>& texture);
    void Absolute(const std::shared_ptr<GLTexture>& texture);

private:
    std::unique_ptr<CudaCore> core_;
    std::map<std::shared_ptr<GLTexture>, std::unique_ptr<GraphicsResource>>
        registerd_textures_;
};

#endif // _CUDA_MAIN_H_