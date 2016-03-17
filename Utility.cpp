#include "Utility.h"

#include <string.h>
#include <math.h>

#include <windows.h>
#include <glsw.h>

#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "multigrid_poisson_solver.h"

using namespace vmath;

static struct {
    GLuint Advect;
    GLuint Jacobi;
    GLuint DampedJacobi;
    GLuint compute_residual;
    GLuint SubtractGradient;
    GLuint ComputeDivergence;
    GLuint ApplyImpulse;
    GLuint ApplyBuoyancy;
} Programs;

const float CellSize = 0.15f;
const int ViewportWidth = 512;
const int GridWidth = 128;
const int ViewportHeight = ViewportWidth;
const int GridHeight = GridWidth;
const int GridDepth = GridWidth;
const float SplatRadius = GridWidth / 24.0f;
const float AmbientTemperature = 0.0f;
const float ImpulseTemperature = 40.0f;
const float ImpulseDensity = 4.0f;
const int NumJacobiIterations = 40;
const int kNumMultigridIterations = 5;
const float kMaxTimeStep = 0.33f;
const float SmokeBuoyancy = 1.0f;
const float SmokeWeight = 0.0001f;
const float GradientScale = 1.125f / CellSize;
const float TemperatureDissipation = 0.95f;
const float VelocityDissipation = 0.999f;
const float DensityDissipation = 0.998f;
const PoissonMethod kSolverChoice = POISSON_SOLVER_MULTI_GRID;
const Vector3 kImpulsePosition(GridWidth / 2.0f, (int)SplatRadius / 2.0f, GridDepth / 2.0f);
const float kBuoyancyCoef = sqrtf(GridWidth / 128.0f);

void CreateObstacles(SurfacePod dest)
{
    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glViewport(0, 0, dest.Width, dest.Height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint program = LoadProgram(FluidShader::GetVertexShaderCode(), 0,
                                 FluidShader::GetFillShaderCode());
    glUseProgram(program);

    GLuint lineVbo;
    glGenBuffers(1, &lineVbo);
    GLuint circleVbo;
    glGenBuffers(1, &circleVbo);
    glEnableVertexAttribArray(SlotPosition);

    for (int slice = 0; slice < dest.Depth; ++slice) {

        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dest.ColorTexture, 0, dest.Depth - 1 - slice);
        float z = dest.Depth / 2.0f;
        z = abs(slice - z) / z;
        float fraction = 1 - sqrt(z);
        float radius = 0.5f * fraction;

        if (slice == 0 || slice == dest.Depth - 1) {
            radius *= 100;
        }

        const bool DrawBorder = true;
        if (DrawBorder && slice != 0 && slice != dest.Depth - 1) {
            #define T 0.9999f
            float positions[] = { -T, -T, T, -T, T,  T, -T,  T, -T, -T };
            #undef T
            GLsizeiptr size = sizeof(positions);
            glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
            glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
            GLsizeiptr stride = 2 * sizeof(positions[0]);
            glVertexAttribPointer(SlotPosition, 2, GL_FLOAT, GL_FALSE, stride, 0);
            glDrawArrays(GL_LINE_STRIP, 0, 5);
        }

        const bool DrawSphere = false;
        if (DrawSphere || slice == 0 || slice == dest.Depth - 1) {
            const int slices = 64;
            float positions[slices*2*3];
            float twopi = 8*atan(1.0f);
            float theta = 0;
            float dtheta = twopi / (float) (slices - 1);
            float* pPositions = &positions[0];
            for (int i = 0; i < slices; i++) {
                *pPositions++ = 0;
                *pPositions++ = 0;

                *pPositions++ = radius * cos(theta);
                *pPositions++ = radius * sin(theta);
                theta += dtheta;

                *pPositions++ = radius * cos(theta);
                *pPositions++ = radius * sin(theta);
            }
            GLsizeiptr size = sizeof(positions);
            glBindBuffer(GL_ARRAY_BUFFER, circleVbo);
            glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
            GLsizeiptr stride = 2 * sizeof(positions[0]);
            glVertexAttribPointer(SlotPosition, 2, GL_FLOAT, GL_FALSE, stride, 0);
            glDrawArrays(GL_TRIANGLES, 0, slices * 3);
        }
    }

    // Cleanup
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &lineVbo);
    glDeleteBuffers(1, &circleVbo);
}

GLuint LoadProgram(const std::string& vs_source, const std::string& gs_source,
                   const std::string& fs_source)
{
    const char* version_directive = "#version 150\n";

    static int first = 1;
    if (first) {
        glswInit();
        glswAddDirective("*", "#version 150");
        first = 0;
    }
    
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
    PezCheckCondition(compileSuccess, "Can't compile vs:\n%s", compilerSpew);
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
        PezCheckCondition(compileSuccess, "Can't compile gs:\n%s", compilerSpew);
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
        PezCheckCondition(compileSuccess, "Can't compile fs:\n%s", compilerSpew);
        glAttachShader(programHandle, fsHandle);
    }

    glBindAttribLocation(programHandle, SlotPosition, "Position");
    glBindAttribLocation(programHandle, SlotTexCoord, "TexCoord");
    glLinkProgram(programHandle);
    
    GLint linkSuccess;
    glGetProgramiv(programHandle, GL_LINK_STATUS, &linkSuccess);
    glGetProgramInfoLog(programHandle, sizeof(compilerSpew), 0, compilerSpew);

    if (!linkSuccess) {
        PezDebugString("Link error.\n");
        PezDebugString("%s\n", compilerSpew);
    }
    
    return programHandle;
}

SurfacePod CreateSurface(GLsizei width, GLsizei height, int numComponents)
{
    GLuint fboHandle;
    glGenFramebuffers(1, &fboHandle);
    glBindFramebuffer(GL_FRAMEBUFFER, fboHandle);

    GLuint textureHandle;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_2D, textureHandle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    switch (numComponents) {
        case 1:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_HALF_FLOAT, 0);
            break;
        case 2:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, width, height, 0, GL_RG, GL_HALF_FLOAT, 0);
            break;
        case 3:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_HALF_FLOAT, 0);
            break;
        case 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_HALF_FLOAT, 0);
            break;
    }

    PezCheckCondition(GL_NO_ERROR == glGetError(), "Unable to create normals texture");

    GLuint colorbuffer;
    glGenRenderbuffers(1, &colorbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandle, 0);
    PezCheckCondition(GL_NO_ERROR == glGetError(), "Unable to attach color buffer");
    
    PezCheckCondition(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER), "Unable to create FBO.");
    SurfacePod surface = { fboHandle, textureHandle };

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    surface.Width = width;
    surface.Height = height;
    surface.Depth = 1;
    return surface;
}

SurfacePod CreateVolume(GLsizei width, GLsizei height, GLsizei depth, int numComponents)
{
    GLuint fboHandle;
    glGenFramebuffers(1, &fboHandle);
    glBindFramebuffer(GL_FRAMEBUFFER, fboHandle);

    GLuint textureHandle;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_3D, textureHandle);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    switch (numComponents) {
        case 1:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, width, height, depth, 0, GL_RED, GL_HALF_FLOAT, 0);
            break;
        case 2:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RG16F, width, height, depth, 0, GL_RG, GL_HALF_FLOAT, 0);
            break;
        case 3:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F, width, height, depth, 0, GL_RGB, GL_HALF_FLOAT, 0);
            break;
        case 4:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, width, height, depth, 0, GL_RGBA, GL_HALF_FLOAT, 0);
            break;
    }

    PezCheckCondition(GL_NO_ERROR == glGetError(), "Unable to create volume texture");

    GLuint colorbuffer;
    glGenRenderbuffers(1, &colorbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, textureHandle, 0);
    PezCheckCondition(GL_NO_ERROR == glGetError(), "Unable to attach color buffer");

    PezCheckCondition(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER), "Unable to create FBO.");
    SurfacePod surface = { fboHandle, textureHandle };

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    surface.Width = width;
    surface.Height = height;
    surface.Depth = depth;
    return surface;
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

void InitSlabOps()
{
    Programs.Advect = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetAdvectShaderCode());
    Programs.Jacobi = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetJacobiShaderCode());
    Programs.DampedJacobi = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetDampedJacobiShaderCode());
    Programs.compute_residual = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), MultigridShader::GetComputeResidualShaderCode());
    Programs.SubtractGradient = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetSubtractGradientShaderCode());
    Programs.ComputeDivergence = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetComputeDivergenceShaderCode());
    Programs.ApplyImpulse = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetSplatShaderCode());
    Programs.ApplyBuoyancy = LoadProgram(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetBuoyancyShaderCode());
}

void ClearSurface(SurfacePod s, float v)
{
    glBindFramebuffer(GL_FRAMEBUFFER, s.FboHandle);
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

void Advect(SurfacePod velocity, SurfacePod source, SurfacePod obstacles, SurfacePod dest, float delta_time, float dissipation)
{
    GLuint p = Programs.Advect;
    glUseProgram(p);

    SetUniform("InverseSize", recipPerElem(Vector3(float(GridWidth), float(GridHeight), float(GridDepth))));
    SetUniform("TimeStep", delta_time);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

void Jacobi(SurfacePod pressure, SurfacePod divergence, SurfacePod obstacles)
{
    GLuint p = Programs.Jacobi;
    glUseProgram(p);

    SetUniform("Alpha", -CellSize * CellSize);
    SetUniform("InverseBeta", 0.1666f);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, pressure.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, pressure.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, divergence.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pressure.Depth);
    ResetState();
}

void DampedJacobi(SurfacePod pressure, SurfacePod divergence,
                  SurfacePod obstacles, float cell_size)
{
    GLuint p = Programs.DampedJacobi;
    glUseProgram(p);

    SetUniform("Alpha", -(cell_size * cell_size));
    SetUniform("InverseBeta", 0.11111111f);
    SetUniform("one_minus_omega", 0.33333333f);
    SetUniform("Pressure", 0);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, pressure.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, pressure.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, divergence.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pressure.Depth);
    ResetState();
}

void SolvePressure(SurfacePod packed)
{
    switch (kSolverChoice) {
        case POISSON_SOLVER_JACOBI:
        case POISSON_SOLVER_GAUSS_SEIDEL: { // Bad in parallelism. Hard to be
                                            // implemented by shader.
//             ClearSurface(pressure, 0.0f);
//             for (int i = 0; i < NumJacobiIterations; ++i) {
//                 Jacobi(pressure, divergence, obstacles);
//             }
            break;
        }
        case POISSON_SOLVER_DAMPED_JACOBI: {
            // NOTE: If we don't clear the buffer, a lot more details are gonna
            //       be rendered. Preconditioned?
            //
            // Our experiments reveals that increasing the iteration times to
            // 80 of Jacobi will NOT lead to higher accuracy.

//             ClearSurface(pressure, 0.0f);
//             for (int i = 0; i < NumJacobiIterations; ++i) {
//                 DampedJacobi(pressure, divergence, obstacles, CellSize);
//             }
            break;
        }
        case POISSON_SOLVER_MULTI_GRID: {
            static PoissonSolver* p_solver = nullptr;
            if (!p_solver) {
                p_solver = new MultigridPoissonSolver();
                p_solver->Initialize(GridWidth);
            }

            // An iteration times lower than 4 will introduce significant
            // unnatural visual effect caused by the half-convergent state of
            // pressure. 
            //
            // If I change the value to 6, then the average |r| could be
            // reduced to around 0.004.
            //
            // And please also note that the average |r| of Damped Jacobi
            // (using a constant iteration time of 40) is stabilized at 0.025.
            // That's a pretty good score!

            for (int i = 0; i < kNumMultigridIterations; i++)
                p_solver->Solve(packed, !i);

            break;
        }
        default: {
            break;
        }
    }
}

void SubtractGradient(SurfacePod velocity, SurfacePod packed)
{
    GLuint p = Programs.SubtractGradient;
    glUseProgram(p);

    SetUniform("GradientScale", GradientScale);
    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("velocity", 0);
    SetUniform("packed_tex", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, velocity.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, packed.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, velocity.Depth);
    ResetState();
}

void ComputeDivergence(SurfacePod velocity, SurfacePod obstacles, SurfacePod dest)
{
    GLuint p = Programs.ComputeDivergence;
    glUseProgram(p);

    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("Obstacles", 1);
    SetUniform("velocity", 0);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

void ApplyImpulse(SurfacePod dest, Vector3 position, Vector3 hotspot, float value_r, float value_g)
{
    GLuint p = Programs.ApplyImpulse;
    glUseProgram(p);

    SetUniform("center_point", position);
    SetUniform("hotspot", hotspot);
    SetUniform("radius", SplatRadius);
    SetUniform("fill_color_r", value_r);
    SetUniform("fill_color_g", value_g);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glEnable(GL_BLEND);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}

void ApplyBuoyancy(SurfacePod velocity, SurfacePod temperature, SurfacePod dest, float delta_time)
{
    GLuint p = Programs.ApplyBuoyancy;
    glUseProgram(p);

    SetUniform("Velocity", 0);
    SetUniform("Temperature", 1);
    SetUniform("AmbientTemperature", AmbientTemperature);
    SetUniform("TimeStep", delta_time);
    SetUniform("Sigma", kBuoyancyCoef);
    SetUniform("Kappa", SmokeWeight);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, temperature.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
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