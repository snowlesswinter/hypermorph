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

int g_diagnosis = 0;

namespace
{
struct
{
    SurfacePod velocity_;
    SurfacePod density_;
    SurfacePod temperature_;
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
    RaycastProgram = LoadProgram(RaycastShader::Vertex(), RaycastShader::Geometry(), RaycastShader::Fragment());
    advect_packed_program_.Load(FluidShader::Vertex(), FluidShader::PickLayer(), FluidShader::GetAdvectPackedShaderCode());
    Vbos.CubeCenter = CreatePointVbo(0, 0, 0);
    Vbos.FullscreenQuad = CreateQuadVbo();

    Surfaces.velocity_ = CreateVolume(GridWidth, GridHeight, GridDepth, 3);

    // A hard lesson had told us: locality is a vital factor of the performance
    // of raycast. Even a trivial-like adjustment that packing the temperature
    // with the density field would surprisingly bring a 17% decline to the
    // performance. 
    //
    // Here is the analysis:
    // 
    // In the original design, density buffer is 128 ^ 3 * 2 byte = 4 MB,
    // where as the buffer had been increased to 128 ^ 3 * 6 byte = 12 MB in
    // our experiment(it is 6 bytes wide instead of 4 because we need to
    // swap it with the 3-byte-width buffer that shared with velocity buffer).
    // That expanded buffer size would greatly increase the possibility of
    // cache miss in GPU during raycast. So, it's a problem all about the cache
    // shortage in graphic cards.

    Surfaces.density_ = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    Surfaces.temperature_ = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
    general_buffers.general_buffer_1 = CreateVolume(GridWidth, GridHeight, GridDepth, 1);
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
    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
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
    static int frame_count = 0;
    frame_count++;

    static double first_time = GetCurrentTimeInSeconds();
    bool render_velocity = GetCurrentTimeInSeconds() - first_time < 10.0;

    if (SimulateFluid) {
        float sin_factor = static_cast<float>(sin(time_elapsed / 4 * Pi));
        float cos_factor = static_cast<float>(cos(time_elapsed / 4 * Pi));
        float hotspot_x =
            cos_factor * SplatRadius * 0.8f + kImpulsePosition.getX();
        float hotspot_z =
            sin_factor * SplatRadius * 0.8f + kImpulsePosition.getZ();
        Vector3 hotspot(hotspot_x, 0, hotspot_z);

        glBindBuffer(GL_ARRAY_BUFFER, Vbos.FullscreenQuad);
        glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);
        glViewport(0, 0, GridWidth, GridHeight);

        Metrics::Instance()->OnFrameUpdateBegins();

        // Advect velocity
        Advect(Surfaces.velocity_, Surfaces.velocity_, SurfacePod(), general_buffers.general_buffer_3, delta_time, VelocityDissipation);
        std::swap(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnVelocityAvected();

        // Advect density and temperature
        ClearSurface(general_buffers.general_buffer_1, 0.0f);
        Advect(Surfaces.velocity_, Surfaces.temperature_, SurfacePod(), general_buffers.general_buffer_1, delta_time, TemperatureDissipation);
        std::swap(Surfaces.temperature_, general_buffers.general_buffer_1);
        Metrics::Instance()->OnTemperatureAvected();

        ClearSurface(general_buffers.general_buffer_1, 0.0f);
        Advect(Surfaces.velocity_, Surfaces.density_, SurfacePod(), general_buffers.general_buffer_1, delta_time, DensityDissipation);
        std::swap(Surfaces.density_, general_buffers.general_buffer_1);
        Metrics::Instance()->OnDensityAvected();

        // Apply buoyancy and gravity
        ApplyBuoyancy(Surfaces.velocity_, Surfaces.temperature_, general_buffers.general_buffer_3, delta_time);
        std::swap(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnBuoyancyApplied();

        // Splat new smoke
        ApplyImpulse(Surfaces.density_, kImpulsePosition, hotspot, ImpulseDensity, ImpulseDensity);
        ApplyImpulse(Surfaces.temperature_, kImpulsePosition, hotspot, ImpulseTemperature, ImpulseTemperature);
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
    ClearSurface(Surfaces.density_, 0.0f);
    ClearSurface(Surfaces.temperature_, 0.0f);
    Metrics::Instance()->Reset();
}

void PezHandleKey(char c)
{
    switch (c) {
        case VK_SPACE:
            SimulateFluid = !SimulateFluid;
            break;
        case 'd':
            Metrics::Instance()->set_diagnosis_mode(
                !Metrics::Instance()->diagnosis_mode());
            break;
        case 'g':
            g_diagnosis = 1 - g_diagnosis;
            break;
        case 'r':
            Reset();
            break;
        case '`':
            track_ball->ReturnHome();
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
