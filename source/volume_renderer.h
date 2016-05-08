#ifndef _VOLUME_RENDERER_H_
#define _VOLUME_RENDERER_H_

#include <memory>

class GLProgram;
class GLSurface;
class GraphicsVolume;
struct MeshPod;
class VolumeRenderer
{
public:
    VolumeRenderer();
    ~VolumeRenderer();

    bool Init(int viewport_width, int viewport_height);
    void OnViewportSized(int viewport_width, int viewport_height);
    void Render(std::shared_ptr<GraphicsVolume> density_volume);

private:
    MeshPod* GetQuadMesh();
    void RenderSurface();

    std::shared_ptr<GLSurface> surf_;
    std::shared_ptr<GLProgram> program_;
    MeshPod* quad_mesh_;
};

#endif // _VOLUME_RENDERER_H_