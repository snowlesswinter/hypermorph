#pragma once
#include <vector>
#include <vmath.hpp>
#include <pez.h>
#include <glew.h>

enum AttributeSlot
{
    SlotPosition,
    SlotTexCoord,
    SlotNormal
};

struct ITrackball
{
    virtual void MouseDown(int x, int y) = 0;
    virtual void MouseUp(int x, int y) = 0;
    virtual void MouseMove(int x, int y) = 0;
    virtual void MouseWheel(int x, int y, float delta) = 0;
    virtual void ReturnHome() = 0;
    virtual vmath::Matrix3 GetRotation() const = 0;
    virtual float GetZoom() const = 0;
    virtual void Update(unsigned int microseconds) = 0;
};

struct TexturePod
{
    GLuint Handle;
    GLsizei Width;
    GLsizei Height;
};

struct SurfacePod
{
    GLuint FboHandle;
    GLuint ColorTexture;
    GLsizei Width;
    GLsizei Height;
    GLsizei Depth;
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

enum PoissonSolver
{
    POISSON_SOLVER_JACOBI,
    POISSON_SOLVER_DAMPED_JACOBI,
    POISSON_SOLVER_GAUSS_SEIDEL,
    POISSON_SOLVER_MULTI_GRID
};

ITrackball* CreateTrackball(float width, float height, float radius);
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
SurfacePod CreateSurface(int width, int height, int numComponents = 4);
SurfacePod CreateVolume(int width, int height, int depth, int numComponents = 4);
GLuint CreatePointVbo(float x, float y, float z);
GLuint CreateQuadVbo();
MeshPod CreateQuadMesh(float left, float top, float right, float bottom);
void CreateObstacles(SurfacePod dest);
void InitSlabOps();
void ClearSurface(SurfacePod s, float v);
void RenderMesh(const MeshPod& mesh);
void Advect(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod dest, float delta_time, float dissipation);
void Jacobi(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles);
void DampedJacobi(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles);
void ComputeResidual(SurfacePod residual, SurfacePod divergence, SurfacePod pressure);
void SolvePressureByMultiGrid(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles);
void SolvePressure(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles);
void SubtractGradient(SurfacePod velocity, SurfacePod pressure, SurfacePod obstacles, SurfacePod dest);
void ComputeDivergence(SurfacePod velocity, SurfacePod obstacles, SurfacePod dest);
void ApplyImpulse(SurfacePod dest, vmath::Vector3 position, vmath::Vector3 hotspot, float value);
void ApplyBuoyancy(SurfacePod velocity, SurfacePod temperature, SurfacePod density, SurfacePod dest, float delta_time);

extern const float CellSize;
extern const int ViewportWidth;
extern const int ViewportHeight;
extern const int GridWidth;
extern const int GridHeight;
extern const int GridDepth;
extern const float SplatRadius;
extern const float AmbientTemperature;
extern const float ImpulseTemperature;
extern const float ImpulseDensity;
extern const int NumJacobiIterations;
extern const float TimeStep;
extern const float SmokeBuoyancy;
extern const float SmokeWeight;
extern const float GradientScale;
extern const float TemperatureDissipation;
extern const float VelocityDissipation;
extern const float DensityDissipation;
extern const vmath::Vector3 kImpulsePosition;
