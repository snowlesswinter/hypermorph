#include "graphics_resource.h"

#include "cuda_core.h"

GraphicsResource::GraphicsResource(CudaCore* core)
    : core_(core)
    , resource_(nullptr)
{

}

GraphicsResource::~GraphicsResource()
{
    if (resource_) {
        core_->UnregisterGLResource(this);
        resource_ = 0;
    }
}

cudaGraphicsResource** GraphicsResource::Receive()
{
    return &resource_;
}
