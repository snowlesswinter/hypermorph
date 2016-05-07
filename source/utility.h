#pragma once

#include <memory>
#include <vector>
#include <vmath.hpp>
#include "third_party/opengl/glew.h"

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
void SetUniform(const char* name, vmath::Matrix4 value);
void SetUniform(const char* name, vmath::Matrix3 value);
void SetUniform(const char* name, vmath::Vector3 value);
void SetUniform(const char* name, vmath::Point3 value);
void SetUniform(const char* name, vmath::Vector4 value);
TexturePod LoadTexture(const char* path);
GLuint CreatePointVbo(float x, float y, float z);
GLuint CreateQuadVbo();
MeshPod CreateQuadMesh(float left, float top, float right, float bottom);
void ClearSurface(GLTexture* s, float v);
void RenderMesh(const MeshPod& mesh);
double GetCurrentTimeInSeconds();
void ResetState();
vmath::Vector3 CalculateInverseSize(const GLVolume& volume);
vmath::Vector3 CalculateInverseSize(const CudaVolume& volume);
void PrintDebugString(const char* content, ...);
void SetFatalError(const char* content, ...);
void CheckCondition(int condition, ...);

extern const float CellSize;
extern const int ViewportWidth;
extern const int ViewportHeight;
extern const int GridWidth;
extern const int GridHeight;
extern const int GridDepth;
extern const float kMaxTimeStep;
extern const float kBuoyancyCoef;
extern const vmath::Vector3 kImpulsePosition;
