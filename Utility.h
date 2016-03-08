#pragma once
#include <vector>
#include <vmath.hpp>
#include <pez.h>
#include <glew.h>

enum AttributeSlot {
    SlotPosition,
    SlotTexCoord,
};

struct ITrackball {
    virtual void MouseDown(int x, int y) = 0;
    virtual void MouseUp(int x, int y) = 0;
    virtual void MouseMove(int x, int y) = 0;
    virtual void ReturnHome() = 0;
    virtual vmath::Matrix3 GetRotation() const = 0;
    virtual float GetZoom() const = 0;
    virtual void Update(unsigned int microseconds) = 0;
};

struct TexturePod {
    GLuint Handle;
    GLsizei Width;
    GLsizei Height;
};

struct SurfacePod {
    GLuint FboHandle;
    GLuint ColorTexture;
    GLsizei Width;
    GLsizei Height;
    GLsizei Depth;
};

struct SlabPod {
    SurfacePod Ping;
    SurfacePod Pong;
};

ITrackball* CreateTrackball(float width, float height, float radius);
GLuint LoadProgram(const char* vsKey, const char* gsKey, const char* fsKey);
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
void CreateObstacles(SurfacePod dest);
SlabPod CreateSlab(GLsizei width, GLsizei height, GLsizei depth, int numComponents);
void InitSlabOps();
void SwapSurfaces(SlabPod* slab);
void ClearSurface(SurfacePod s, float v);
void Advect(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod dest, float delta_time, float dissipation);
void Jacobi(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles, SurfacePod dest);
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
extern const vmath::Vector3 impulse_position;
