#include "cuda_core.h"

#include <cassert>

#include "glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "opengl/gl_texture.h"
#include "graphics_resource.h"

CudaCore::CudaCore()
{

}

CudaCore::~CudaCore()
{
    cudaDeviceReset();
}

bool CudaCore::Init()
{
    findCudaGLDevice(0, nullptr);
    return 0;
}

int CudaCore::RegisterGLImage(const GLTexture& texture,
                              GraphicsResource* graphics_res)
{
    cudaError_t result = cudaGraphicsGLRegisterImage(
        graphics_res->Receive(), texture.handle(), texture.target(),
        cudaGraphicsRegisterFlagsNone);
    assert(result == cudaSuccess);
    return result == cudaSuccess ? 0 : -1;
}

void CudaCore::UnregisterGLImage(GraphicsResource* graphics_res)
{
    cudaGraphicsUnregisterResource(graphics_res->resource());
}
