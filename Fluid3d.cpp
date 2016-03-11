#include "Utility.h"
#include <cmath>
#include <algorithm>
#include <sstream>

#include "raycast_shader.h"
#include "overlay_content.h"
#include "metrics.h"

using namespace vmath;
using std::string;

namespace
{
struct
{
    SurfacePod Velocity;
    SurfacePod density_;
    SurfacePod temperature_;
    SurfacePod Pressure;
} Surfaces;

struct
{
    SurfacePod general_buffer_1;
    SurfacePod general_buffer_3;
} general_buffers;

struct
{
    Matrix4 Projection;
    Matrix4 Modelview;
    Matrix4 View;
    Matrix4 ModelviewProjection;
} Matrices;

struct
{
    GLuint CubeCenter;
    GLuint FullscreenQuad;
} Vbos;

ITrackball* track_ball;
Point3 EyePosition;
GLuint RaycastProgram;
float FieldOfView = 0.7f;
bool SimulateFluid = true;
OverlayContent overlay_;
Metrics metrics_;
bool measure_performance_ = false;
}

PezConfig PezGetConfig()
{
    static PezConfig config;
    config.Title = "Fluid3d";
    config.Width = ViewportWidth;
    config.Height = ViewportHeight;
    config.Multisampling = 0;
    config.VerticalSync = 0;
    return config;
}

void PezInitialize()
{
    PezConfig cfg = PezGetConfig();

    track_ball = CreateTrackball(cfg.Width * 1.0f, cfg.Height * 1.0f, cfg.Width * 0.5f);
    RaycastProgram = LoadProgram(RaycastShader::GetVertexShaderCode(), RaycastShader::GetGeometryShaderCode(), RaycastShader::GetFragmentShaderCode());
    Vbos.CubeCenter = CreatePointVbo(0, 0, 0);
    Vbos.FullscreenQuad = CreateQuadVbo();

    Surfaces.Velocity = CreateVolume(GridWidth, GridHeight, GridDepth, 3);
    Surfaces.density_ = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    Surfaces.temperature_ = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    Surfaces.Pressure = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    general_buffers.general_buffer_1 = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    general_buffers.general_buffer_3 = CreateVolume(GridWidth, GridHeight, GridDepth, 3);
    InitSlabOps();
    ClearSurface(Surfaces.temperature_, AmbientTemperature);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableVertexAttribArray(SlotPosition);
}

void DisplayMetrics()
{
    std::stringstream text;
    text.precision(2);
    text << std::fixed << metrics_.GetFrameRate() << " f/s" << std::endl;
    char* o[] = {
        "Velocity",
        "Temperature",
        "Density",
        "Buoyancy",
        "Impulse",
        "Divergence",
        "Pressure",
        "Gradient",
        "Raycast",
    };
    for (int i = 0; i < sizeof(o) / sizeof(o[0]); i++) {
        float cost = metrics_.GetOperationTimeCost(
            static_cast<Metrics::Operations>(i));
        text << o[i] << ": " << cost << std::endl;
    }

    overlay_.RenderText(text.str());
}

void PezRender()
{
    if (measure_performance_) {
        glFinish();
        metrics_.OnFrameRenderingBegins(GetCurrentTimeInSeconds());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    PezConfig cfg = PezGetConfig();
    glViewport(0, 0, cfg.Width, cfg.Height);
    glClearColor(0.01f, 0.06f, 0.08f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBindBuffer(GL_ARRAY_BUFFER, Vbos.CubeCenter);
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glBindTexture(GL_TEXTURE_3D, Surfaces.density_.ColorTexture);
    glUseProgram(RaycastProgram);
    SetUniform("ModelviewProjection", Matrices.ModelviewProjection);
    SetUniform("Modelview", Matrices.Modelview);
    SetUniform("ViewMatrix", Matrices.View);
    SetUniform("ProjectionMatrix", Matrices.Projection);
    SetUniform("RayStartPoints", 1);
    SetUniform("RayStopPoints", 2);
    SetUniform("EyePosition", EyePosition);
    SetUniform("RayOrigin", Vector4(transpose(Matrices.Modelview) * EyePosition).getXYZ());
    SetUniform("FocalLength", 1.0f / std::tan(FieldOfView / 2));
    SetUniform("WindowSize", float(cfg.Width), float(cfg.Height));
    glDrawArrays(GL_POINTS, 0, 1);

    if (measure_performance_)
        glFinish();

    metrics_.OnFrameRendered(GetCurrentTimeInSeconds());

    DisplayMetrics();
}

void PezUpdate(unsigned int microseconds)
{
    float dt = microseconds * 0.000001f;
    track_ball->Update(microseconds);
    EyePosition = Point3(0, 0, 3.5f + track_ball->GetZoom());
    Vector3 up(0, 1, 0); Point3 target(0);
    Matrices.View = Matrix4::lookAt(EyePosition, target, up);
    Matrix4 modelMatrix(transpose(track_ball->GetRotation()), Vector3(0));
    modelMatrix *= Matrix4::rotationY(0.5f);
    Matrices.Modelview = Matrices.View * modelMatrix;
    
    Matrices.Projection = Matrix4::perspective(
        FieldOfView,
        float(ViewportWidth) / ViewportHeight, // Aspect Ratio
        0.0f,   // Near Plane
        1.0f);  // Far Plane

    Matrices.ModelviewProjection = Matrices.Projection * Matrices.Modelview;
    static double time_elapsed = 0;
    time_elapsed += dt;

    // Use constant time step. The reason is explained in the fluid shader.
    //
    // Note: The behaviors of fluid are visually equivalent for 
    //       constant/non-constant time steps(need to modify the coefficient
    //       of buoyancy formula to keep a high acceleration to voxel).

    float delta_time = kMaxTimeStep;

    if (SimulateFluid) {
        double hotspot_x = cos(time_elapsed * Pi) * SplatRadius * 0.8 +
            kImpulsePosition.getX();
        double hotspot_z = sin(time_elapsed * Pi) * SplatRadius * 0.8 +
            kImpulsePosition.getZ();
        Vector3 hotspot(static_cast<float>(hotspot_x), 0,
                        static_cast<float>(hotspot_z));

        glBindBuffer(GL_ARRAY_BUFFER, Vbos.FullscreenQuad);
        glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);
        glViewport(0, 0, GridWidth, GridHeight);

        if (measure_performance_) {
            glFinish();
            metrics_.OnFrameUpdateBegins(GetCurrentTimeInSeconds());
        }

        // Advect velocity
        Advect(Surfaces.Velocity, Surfaces.Velocity, SurfacePod(), general_buffers.general_buffer_3, delta_time, VelocityDissipation);
        std::swap(Surfaces.Velocity, general_buffers.general_buffer_3);
        if (measure_performance_) {
            glFinish();
            metrics_.OnVelocityAvected(GetCurrentTimeInSeconds());
        }

        // Advect temperature
        ClearSurface(general_buffers.general_buffer_1, 0);
        Advect(Surfaces.Velocity, Surfaces.temperature_, SurfacePod(), general_buffers.general_buffer_1, delta_time, TemperatureDissipation);
        std::swap(Surfaces.temperature_, general_buffers.general_buffer_1);
        if (measure_performance_) {
            glFinish();
            metrics_.OnTemperatureAvected(GetCurrentTimeInSeconds());
        }

        // Advect density
        ClearSurface(general_buffers.general_buffer_1, 0);
        Advect(Surfaces.Velocity, Surfaces.density_, SurfacePod(), general_buffers.general_buffer_1, delta_time, DensityDissipation);
        std::swap(Surfaces.density_, general_buffers.general_buffer_1);
        if (measure_performance_) {
            glFinish();
            metrics_.OnDensityAvected(GetCurrentTimeInSeconds());
        }

        // Apply buoyancy and gravity
        ApplyBuoyancy(Surfaces.Velocity, Surfaces.temperature_, general_buffers.general_buffer_3, delta_time);
        std::swap(Surfaces.Velocity, general_buffers.general_buffer_3);
        if (measure_performance_) {
            glFinish();
            metrics_.OnBuoyancyApplied(GetCurrentTimeInSeconds());
        }

        // Splat new smoke
        ApplyImpulse(Surfaces.temperature_, kImpulsePosition, hotspot, ImpulseTemperature);
        ApplyImpulse(Surfaces.density_, kImpulsePosition, hotspot, ImpulseDensity);
        if (measure_performance_) {
            glFinish();
            metrics_.OnImpulseApplied(GetCurrentTimeInSeconds());
        }

        // Calculate divergence
        ClearSurface(general_buffers.general_buffer_1, 0);

        // TODO: Try to slightly optimize the calculation by pre-multiplying 1/h^2.
        ComputeDivergence(Surfaces.Velocity, SurfacePod(), general_buffers.general_buffer_1);
        if (measure_performance_) {
            glFinish();
            metrics_.OnDivergenceComputed(GetCurrentTimeInSeconds());
        }

        // Solve pressure-velocity Poisson equation
        SolvePressure(Surfaces.Pressure, general_buffers.general_buffer_1, SurfacePod());
        if (measure_performance_) {
            glFinish();
            metrics_.OnPressureSolved(GetCurrentTimeInSeconds());
        }

        // Rectify velocity via the gradient of pressure
        SubtractGradient(Surfaces.Velocity, Surfaces.Pressure, SurfacePod(), general_buffers.general_buffer_3);
        std::swap(Surfaces.Velocity, general_buffers.general_buffer_3);
        if (measure_performance_) {
            glFinish();
            metrics_.OnVelocityRectified(GetCurrentTimeInSeconds());
        }
    }
}

void PezHandleMouse(int x, int y, int action, int delta)
{
    if (action & PEZ_DOWN) track_ball->MouseDown(x, y);
    else if (action & PEZ_UP) track_ball->MouseUp(x, y);
    else if (action & PEZ_MOVE) track_ball->MouseMove(x, y);
    else if (action & PEZ_DOUBLECLICK) track_ball->ReturnHome();

    if (action & PEZ_DOWN) {
        track_ball->MouseDown(x, y);
    } else if (action & PEZ_UP) {
        track_ball->MouseUp(x, y);
    } else if (action & PEZ_MOVE) {
        track_ball->MouseMove(x, y);
    } else if (action & PEZ_WHEEL) {
        float d = (float)delta / 1000 * std::max(abs(track_ball->GetZoom()), 1.0f);
        track_ball->MouseWheel(x, y, -d);
    } else if (action & PEZ_DOUBLECLICK) {
        track_ball->ReturnHome();
    }
}

void PezHandleKey(char c)
{
    SimulateFluid = !SimulateFluid;
}
