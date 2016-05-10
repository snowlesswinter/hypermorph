#ifndef _VOLUME_RENDERER_H_
#define _VOLUME_RENDERER_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

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
    void Raycast(std::shared_ptr<GraphicsVolume> density_volume,
                 const glm::mat4& model_view, const glm::vec3& eye_pos,
                 float focal_length);
    void Render();

private:
    MeshPod* GetQuadMesh();

    std::shared_ptr<GLSurface> surf_;
    std::shared_ptr<GLProgram> render_texture_;
    MeshPod* quad_mesh_;
};

#endif // _VOLUME_RENDERER_H_