#ifndef _GRAPHICS_VOLUME_H_
#define _GRAPHICS_VOLUME_H_

#include <memory>

#include "graphics_lib_enum.h"

class GLTexture;
class CudaVolume;
class GraphicsVolume
{
public:
    explicit GraphicsVolume(GraphicsLib lib);
    ~GraphicsVolume();

    void Clear();
    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width);
    void Reset(GraphicsLib lib);

    int GetWidth() const;
    int GetHeight() const;
    int GetDepth() const;

    std::shared_ptr<GLTexture> gl_texture() const;
    std::shared_ptr<CudaVolume> cuda_volume() const;

private:
    GraphicsLib graphics_lib_;
    std::shared_ptr<GLTexture> gl_texture_;
    std::shared_ptr<CudaVolume> cuda_volume_;
};

#endif // _GRAPHICS_VOLUME_H_