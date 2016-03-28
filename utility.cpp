#include "stdafx.h"
#include "utility.h"

#include <string.h>
#include <math.h>

#include <windows.h>

#include "opengl/gl_texture.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "cuda_host/cuda_volume.h"

using namespace vmath;

const float CellSize = 0.15f; // By far I hadn't figured out how the cell size
                             // should be transformed between levels in
                             // Multigrid V-Cycle. So I just temporarily
                             // discard the original value 0.15f, and simply
                             // set it to 1.0f. It seems that a huge
                             // improvement in quality, but actually it's just
                             // a change on |r|'s scale.
const int ViewportWidth = 512;
const int GridWidth = 128;
const int ViewportHeight = ViewportWidth;
const int GridHeight = GridWidth;
const int GridDepth = GridWidth;
const float SplatRadius = GridWidth / 4.0f;
const float AmbientTemperature = 0.0f;
const float ImpulseTemperature = 40.0f;
const float ImpulseDensity = 3.0f;
const int kNumJacobiIterations = 40;
const float kMaxTimeStep = 0.33f;
const float SmokeBuoyancy = 1.0f;
const float SmokeWeight = 0.0001f;
const float GradientScale = 1.125f / CellSize;
const float TemperatureDissipation = 0.95f;
const float VelocityDissipation = 0.999f;
const float DensityDissipation = 0.988f;
const Vector3 kImpulsePosition(GridWidth / 2.0f, (int)SplatRadius / 2.0f, GridDepth / 2.0f);
const float kBuoyancyCoef = sqrtf(GridWidth / 128.0f);

GLuint LoadProgram(const std::string& vs_source, const std::string& gs_source,
                   const std::string& fs_source)
{
    const char* version_directive = "#version 150\n";
    
    GLint compileSuccess;
    GLchar compilerSpew[256];
    GLuint programHandle = glCreateProgram();

    GLuint vsHandle = glCreateShader(GL_VERTEX_SHADER);
    std::string source = version_directive + vs_source;
    const GLchar* s = source.c_str();
    glShaderSource(vsHandle, 1, &s, 0);
    glCompileShader(vsHandle);
    glGetShaderiv(vsHandle, GL_COMPILE_STATUS, &compileSuccess);
    glGetShaderInfoLog(vsHandle, sizeof(compilerSpew), 0, compilerSpew);
    CheckCondition(compileSuccess, "Can't compile vs:\n%s", compilerSpew);
    glAttachShader(programHandle, vsHandle);

    GLuint gsHandle;
    if (!gs_source.empty())
    {
        gsHandle = glCreateShader(GL_GEOMETRY_SHADER);
        source = version_directive + gs_source;
        const GLchar* s = source.c_str();
        glShaderSource(gsHandle, 1, &s, 0);
        glCompileShader(gsHandle);
        glGetShaderiv(gsHandle, GL_COMPILE_STATUS, &compileSuccess);
        glGetShaderInfoLog(gsHandle, sizeof(compilerSpew), 0, compilerSpew);
        CheckCondition(compileSuccess, "Can't compile gs:\n%s", compilerSpew);
        glAttachShader(programHandle, gsHandle);
    }
    
    GLuint fsHandle;
    if (!fs_source.empty()) {
        fsHandle = glCreateShader(GL_FRAGMENT_SHADER);
        source = version_directive + fs_source;
        const GLchar* s = source.c_str();
        glShaderSource(fsHandle, 1, &s, 0);
        glCompileShader(fsHandle);
        glGetShaderiv(fsHandle, GL_COMPILE_STATUS, &compileSuccess);
        glGetShaderInfoLog(fsHandle, sizeof(compilerSpew), 0, compilerSpew);
        CheckCondition(compileSuccess, "Can't compile fs:\n%s", compilerSpew);
        glAttachShader(programHandle, fsHandle);
    }

    glBindAttribLocation(programHandle, SlotPosition, "Position");
    glBindAttribLocation(programHandle, SlotTexCoord, "TexCoord");
    glLinkProgram(programHandle);
    
    GLint linkSuccess;
    glGetProgramiv(programHandle, GL_LINK_STATUS, &linkSuccess);
    glGetProgramInfoLog(programHandle, sizeof(compilerSpew), 0, compilerSpew);

    if (!linkSuccess) {
        PrintDebugString("Link error.\n");
        PrintDebugString("%s\n", compilerSpew);
    }
    
    return programHandle;
}

void ResetState()
{
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glDisable(GL_BLEND);
}

void ClearSurface(GLTexture* s, float v)
{
    glBindFramebuffer(GL_FRAMEBUFFER, s->frame_buffer());
    glClearColor(v, v, v, v);
    glClear(GL_COLOR_BUFFER_BIT);
}

void RenderMesh(const MeshPod& mesh)
{
    glBindBuffer(GL_ARRAY_BUFFER, mesh.positions_buffer_);
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(SlotPosition);

    if (mesh.coords_buffer_)
    {
        glBindBuffer(GL_ARRAY_BUFFER, mesh.coords_buffer_);
        glVertexAttribPointer(SlotTexCoord, 2, GL_FLOAT, GL_FALSE,
                              2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(SlotTexCoord);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.index_buffer_);
    glDrawElements(GL_TRIANGLES, mesh.index_count_, GL_UNSIGNED_INT, nullptr);
}

GLuint CreatePointVbo(float x, float y, float z)
{
    float p[] = {x, y, z};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(p), &p[0], GL_STATIC_DRAW);
    return vbo;
}

void SetUniform(const char* name, int value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform1i(location, value);
}

void SetUniform(const char* name, float value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform1f(location, value);
}

void SetUniform(const char* name, Matrix4 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniformMatrix4fv(location, 1, 0, (float*) &value);
}

void SetUniform(const char* name, Matrix3 nm)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    float packed[9] = {
        nm.getRow(0).getX(), nm.getRow(1).getX(), nm.getRow(2).getX(),
        nm.getRow(0).getY(), nm.getRow(1).getY(), nm.getRow(2).getY(),
        nm.getRow(0).getZ(), nm.getRow(1).getZ(), nm.getRow(2).getZ() };
    glUniformMatrix3fv(location, 1, 0, &packed[0]);
}

void SetUniform(const char* name, Vector3 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform3f(location, value.getX(), value.getY(), value.getZ());
}

void SetUniform(const char* name, float x, float y)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform2f(location, x, y);
}

void SetUniform(const char* name, Vector4 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform4f(location, value.getX(), value.getY(), value.getZ(), value.getW());
}

void SetUniform(const char* name, Point3 value)
{
    GLuint program;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*) &program);
    GLint location = glGetUniformLocation(program, name);
    glUniform3f(location, value.getX(), value.getY(), value.getZ());
}

GLuint CreateQuadVbo()
{
    short positions[] = {
        -1, -1,
         1, -1,
        -1,  1,
         1,  1,
    };
    GLuint vbo;
    GLsizeiptr size = sizeof(positions);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
    return vbo;
}

MeshPod CreateQuadMesh(float left, float top, float right, float bottom)
{
    MeshPod pod;
    pod.vertex_count_ = 4;
    pod.index_count_ = 6;

    float positions[] = {
        left, top, 0,
        right, top, 0,
        right, bottom, 0,
        left, bottom, 0,
    };

    glGenBuffers(1, &pod.positions_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, pod.positions_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

    float texture_coords[] = {
        0, 0,
        1, 0,
        1, 1,
        0, 1,
    };

    glGenBuffers(1, &pod.coords_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, pod.coords_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texture_coords), texture_coords,
                 GL_STATIC_DRAW);

    int faces[] = {3, 2, 1, 1, 0, 3};
    int facesAnti[] = {0, 1, 2, 0, 2, 3};

    int* p = faces;
    if (top < bottom)
        p = facesAnti;

    glGenBuffers(1, &pod.index_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pod.index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faces), p, GL_STATIC_DRAW);

    return pod;
}

double GetCurrentTimeInSeconds()
{
    static LARGE_INTEGER freqTime = {};
    if (!freqTime.QuadPart)
        QueryPerformanceFrequency(&freqTime);

    LARGE_INTEGER currentTime;
    QueryPerformanceCounter(&currentTime);
    return static_cast<double>(currentTime.QuadPart) / freqTime.QuadPart;
}

vmath::Vector3 CalculateInverseSize(const GLTexture& volume)
{
    return recipPerElem(
        vmath::Vector3(static_cast<float>(volume.width()),
                       static_cast<float>(volume.height()),
                       static_cast<float>(volume.depth())));
}

vmath::Vector3 CalculateInverseSize(const CudaVolume& volume)
{
    return recipPerElem(
        vmath::Vector3(static_cast<float>(volume.width()),
                       static_cast<float>(volume.height()),
                       static_cast<float>(volume.depth())));
}

void PrintDebugString(const char* content, ...)
{
    char msg[1024] = {0};

    va_list a;
    va_start(a, content);

    _vsnprintf_s(msg, _countof(msg), _TRUNCATE, content, a);
    OutputDebugStringA(msg);
}

void FatalErrorImpl(const char* content, va_list a)
{
    char msg[1024] = {0};

    _vsnprintf_s(msg, _countof(msg), _TRUNCATE, content, a);
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
    exit(1);
}

void SetFatalError(const char* content, ...)
{
    va_list a;
    va_start(a, content);
    FatalErrorImpl(content, a);
}

void CheckCondition(int condition, ...)
{
    va_list a;
    const char* content;

    if (condition)
        return;

    va_start(a, condition);
    content = va_arg(a, const char*);
    FatalErrorImpl(content, a);
}