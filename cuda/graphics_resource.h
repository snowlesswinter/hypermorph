#ifndef _GRAPHICS_RESOURCE_H_
#define _GRAPHICS_RESOURCE_H_

struct cudaGraphicsResource;
class CudaCore;
class GraphicsResource
{
public:
    explicit GraphicsResource(CudaCore* core);
    ~GraphicsResource();

    cudaGraphicsResource** Receive();
    cudaGraphicsResource* resource() const { return resource_; }

private:
    CudaCore* core_;
    cudaGraphicsResource* resource_;
};

#endif // _GRAPHICS_RESOURCE_H_