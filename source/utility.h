#pragma once

#include <memory>
#include <vector>

#include "third_party/opengl/glew.h"
#include "third_party/glm/fwd.hpp"

class GLTexture;
class GLVolume;
class CudaVolume;
enum AttributeSlot
{
    SlotPosition,
    SlotTexCoord,
    SlotNormal
};

struct TexturePod
{
    GLuint Handle;
    GLsizei Width;
    GLsizei Height;
};

struct MeshPod
{
public:
    GLuint index_buffer_;
    GLuint positions_buffer_;
    GLuint coords_buffer_;
    GLsizei index_count_;
    GLsizei vertex_count_;
};

GLuint LoadProgram(const std::string& vs_source, const std::string& gs_source, const std::string& fs_source);
void SetUniform(const char* name, int value);
void SetUniform(const char* name, float value);
void SetUniform(const char* name, float x, float y);
void SetUniform(const char* name, glm::mat4 value);
void SetUniform(const char* name, glm::vec3 value);
TexturePod LoadTexture(const char* path);
GLuint CreatePointVbo(float x, float y, float z);
GLuint CreateQuadVbo();
MeshPod CreateQuadMesh(float left, float top, float right, float bottom);
void RenderMesh(const MeshPod& mesh);
void ClearSurface(GLTexture* s, float v);
double GetCurrentTimeInSeconds();
void ResetState();
glm::vec3 CalculateInverseSize(const GLVolume& volume);
glm::vec3 CalculateInverseSize(const CudaVolume& volume);
void PrintDebugString(const char* content, ...);
void SetFatalError(const char* content, ...);
void CheckCondition(int condition, ...);
