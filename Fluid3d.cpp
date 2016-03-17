#include "fluid3d.h"

#include <cmath>
#include <algorithm>
#include <sstream>

#include <windows.h>

#include "gl_program.h"
#include "shader/fluid_shader.h"
#include "shader/raycast_shader.h"
#include "overlay_content.h"
#include "metrics.h"
#include "utility.h"

using namespace vmath;
using std::string;

namespace
{
struct
{
    SurfacePod velocity_;
    SurfacePod density_and_temperature_;
} Surfaces;

struct
{
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
GLProgram advect_packed_program_;
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
    advect_packed_program_.Load(FluidShader::GetVertexShaderCode(), FluidShader::GetPickLayerShaderCode(), FluidShader::GetAdvectPackedShaderCode());
    Vbos.CubeCenter = CreatePointVbo(0, 0, 0);
    Vbos.FullscreenQuad = CreateQuadVbo();

    Surfaces.velocity_ = CreateVolume(GridWidth, GridHeight, GridDepth, 3);
    Surfaces.density_and_temperature_ = CreateVolume(GridWidth, GridHeight, GridDepth, 3);
    general_buffers.general_buffer_3 = CreateVolume(GridWidth, GridHeight, GridDepth, 3);
    InitSlabOps();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableVertexAttribArray(SlotPosition);

    Metrics::Instance()->SetOperationSync([]() { glFinish(); });
    Metrics::Instance()->SetTimeSource(
        []() -> double { return GetCurrentTimeInSeconds(); });
}

void DisplayMetrics()
{
    std::stringstream text;
    text.precision(2);
    text << std::fixed << Metrics::Instance()->GetFrameRate() << " f/s" <<
        std::endl;
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
        "Prolongate",
    };
    for (int i = 0; i < sizeof(o) / sizeof(o[0]); i++) {
        float cost = Metrics::Instance()->GetOperationTimeCost(
            static_cast<Metrics::Operations>(i));
        if (cost > 0.01f)
            text << o[i] << ": " << cost << std::endl;
    }

    overlay_.RenderText(text.str());
}

void PezRender()
{
    Metrics::Instance()->OnFrameRenderingBegins();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    PezConfig cfg = PezGetConfig();
    glViewport(0, 0, cfg.Width, cfg.Height);
    glClearColor(0.01f, 0.06f, 0.08f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBindBuffer(GL_ARRAY_BUFFER, Vbos.CubeCenter);
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glBindTexture(GL_TEXTURE_3D, Surfaces.density_and_temperature_.ColorTexture);
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

    Metrics::Instance()->OnFrameRendered();

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

        Metrics::Instance()->OnFrameUpdateBegins();

        // Advect velocity
        Advect(Surfaces.velocity_, Surfaces.velocity_, SurfacePod(), general_buffers.general_buffer_3, delta_time, VelocityDissipation);
        std::swap(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnVelocityAvected();

        // Advect density and temperature
        AdvectPacked(Surfaces.velocity_, Surfaces.density_and_temperature_,
                     general_buffers.general_buffer_3, delta_time,
                     DensityDissipation, TemperatureDissipation);
        std::swap(Surfaces.density_and_temperature_, general_buffers.general_buffer_3);
        //Metrics::Instance()->OnTemperatureAvected();
        Metrics::Instance()->OnDensityAvected();

        // Apply buoyancy and gravity
        ApplyBuoyancy(Surfaces.velocity_, Surfaces.density_and_temperature_, general_buffers.general_buffer_3, delta_time);
        std::swap(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnBuoyancyApplied();

        // Splat new smoke
        ApplyImpulse(Surfaces.density_and_temperature_, kImpulsePosition, hotspot, ImpulseDensity, ImpulseTemperature);
        Metrics::Instance()->OnImpulseApplied();

        // TODO: Try to slightly optimize the calculation by pre-multiplying 1/h^2.
        ComputeDivergence(Surfaces.velocity_, SurfacePod(),
                          general_buffers.general_buffer_3);
        Metrics::Instance()->OnDivergenceComputed();

        // Solve pressure-velocity Poisson equation
        SolvePressure(general_buffers.general_buffer_3);
        Metrics::Instance()->OnPressureSolved();

        // Rectify velocity via the gradient of pressure
        SubtractGradient(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnVelocityRectified();
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

void Reset()
{
    ClearSurface(Surfaces.velocity_, 0.0f);
    ClearSurface(Surfaces.density_and_temperature_, 0.0f);
    Metrics::Instance()->Reset();
}

void PezHandleKey(char c)
{
    switch (c) {
        case VK_SPACE:
            SimulateFluid = !SimulateFluid;
            break;
        case 'D':
            Metrics::Instance()->set_diagnosis_mode(
                !Metrics::Instance()->diagnosis_mode());
            break;
        case 'R':
            Reset();
            break;
    }
    
}

void AdvectPacked(SurfacePod velocity, SurfacePod source, SurfacePod dest,
                  float delta_time, float dissipation_r, float dissipation_g)
{
    advect_packed_program_.Use();

    SetUniform("velocity", 0);
    SetUniform("source", 1);
    SetUniform("inverse_size", recipPerElem(Vector3(float(GridWidth), float(GridHeight), float(GridDepth))));
    SetUniform("time_step", delta_time);
    SetUniform("dissipation_r", dissipation_r);
    SetUniform("dissipation_g", dissipation_g);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}
