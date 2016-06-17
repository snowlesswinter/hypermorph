#ifndef _GRAPHICS_VOLUME_H_
#define _GRAPHICS_VOLUME_H_

#include <memory>

#include "graphics_lib_enum.h"

class CudaVolume;
class GLVolume;
class GraphicsVolume
{
public:
    explicit GraphicsVolume(GraphicsLib lib);
    ~GraphicsVolume();

    void Clear();
    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width, int border);
    bool HasSameProperties(const GraphicsVolume& other) const;

    GraphicsLib graphics_lib() const { return graphics_lib_; }
    int GetWidth() const;
    int GetHeight() const;
    int GetDepth() const;
    int GetByteWidth() const;

    std::shared_ptr<GLVolume> gl_volume() const;
    std::shared_ptr<CudaVolume> cuda_volume() const;

private:
    GraphicsLib graphics_lib_;
    std::shared_ptr<GLVolume> gl_volume_;
    std::shared_ptr<CudaVolume> cuda_volume_;
    int border_;
};

#endif // _GRAPHICS_VOLUME_H_