#include "cuda_main.h"

#include "glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "opengl/gl_texture.h"

CudaMain* CudaMain::Instance()
{
    static CudaMain* instance = nullptr;
    if (!instance)
        instance = new CudaMain();

    return instance;
}

CudaMain::CudaMain()
    : graphics_res_(nullptr)
{

}

CudaMain::~CudaMain()
{

}

int CudaMain::RegisterGLImage(const GLTexture& texture)
{
    cudaError_t result = cudaGraphicsGLRegisterImage(
        &graphics_res_, texture.handle(), texture.target(),
        cudaGraphicsRegisterFlagsWriteDiscard);
    return result == cudaSuccess ? 0 : -1;
}
