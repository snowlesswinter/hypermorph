#ifndef _VOLUME_RENDERER_H_
#define _VOLUME_RENDERER_H_

#include <memory>

#include "graphics_lib_enum.h"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

class GLProgram;
class GLSurface;
class GraphicsVolume;
struct MeshPod;
class VolumeRenderer
{
public:
    VolumeRenderer();
    ~VolumeRenderer();

    bool Init(const glm::ivec2& viewport_size);
    void OnViewportSized(const glm::ivec2& viewport_size);
    void Render(std::shared_ptr<GraphicsVolume> density_volume,
                float focal_length);
    void Update(glm::vec3 eye_position, const glm::mat4& look_at,
                const glm::mat4& rotation, const glm::mat4& perspective);

    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }

private:
    uint32_t GetCubeCenterVbo();
    MeshPod* GetQuadMesh();
    GLProgram* GetRaycastProgram();

    void RenderImplCuda();
    void RenderImplGlsl(std::shared_ptr<GraphicsVolume> density_volume,
                        float focal_length);

    GraphicsLib graphics_lib_;
    glm::ivec2 viewport_size_;

    glm::mat4 model_view_;
    glm::mat4 view_;
    glm::mat4 projection_;
    glm::mat4 model_view_projection_;
    glm::vec3 eye_position_;

    std::shared_ptr<GLSurface> surf_;
    std::shared_ptr<GLProgram> render_texture_;
    std::shared_ptr<GLProgram> raycast_;
    MeshPod* quad_mesh_;
    uint32_t cube_center_vbo_;

};

#endif // _VOLUME_RENDERER_H_