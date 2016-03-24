#include "fluid3d.h"

#include <cmath>
#include <algorithm>
#include <sstream>

#include <windows.h>

#include "metrics.h"
#include "opengl/gl_program.h"
#include "overlay_content.h"
#include "shader/fluid_shader.h"
#include "shader/raycast_shader.h"
#include "utility.h"

// TODO
#include "multigrid_core.h"
#include "opengl/gl_texture.h"
#include "multigrid_poisson_solver.h"
#include "cuda/cuda_main.h"

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
    std::shared_ptr<GLTexture>* tex_velocity;
    std::shared_ptr<GLTexture>* tex_density;
    std::shared_ptr<GLTexture>* tex_temperature;
} Surfaces;

struct
{
    SurfacePod general_buffer_1;
    SurfacePod general_buffer_3;
} general_buffers;

std::shared_ptr<GLTexture>* gb1;
std::shared_ptr<GLTexture>* gb3;

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
    int c[5] = {0};
    PezConfig cfg = PezGetConfig();

    track_ball = CreateTrackball(cfg.Width * 1.0f, cfg.Height * 1.0f, cfg.Width * 0.5f);
    RaycastProgram = LoadProgram(RaycastShader::Vertex(), RaycastShader::Geometry(), RaycastShader::Fragment());
    advect_packed_program_.Load(FluidShader::Vertex(), FluidShader::PickLayer(), FluidShader::GetAdvectPackedShaderCode());
    Vbos.CubeCenter = CreatePointVbo(0, 0, 0);
    Vbos.FullscreenQuad = CreateQuadVbo();

    MultigridCore core;
    Surfaces.tex_velocity = new std::shared_ptr <GLTexture>();
    *Surfaces.tex_velocity = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_RGBA32F, GL_RGBA);


    SurfacePod kk;
    kk.FboHandle = (*Surfaces.tex_velocity)->frame_buffer();
    kk.ColorTexture = (*Surfaces.tex_velocity)->handle();
    kk.Width = (*Surfaces.tex_velocity)->width();
    kk.Height = (*Surfaces.tex_velocity)->height();
    kk.Depth = (*Surfaces.tex_velocity)->depth();

    Surfaces.velocity_ = kk;

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

    Surfaces.tex_density = new std::shared_ptr <GLTexture>();
    *Surfaces.tex_density = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_R32F, GL_RED);

    kk.FboHandle = (*Surfaces.tex_density)->frame_buffer();
    kk.ColorTexture = (*Surfaces.tex_density)->handle();
    kk.Width = (*Surfaces.tex_density)->width();
    kk.Height = (*Surfaces.tex_density)->height();
    kk.Depth = (*Surfaces.tex_density)->depth();

    Surfaces.density_ = kk;

    Surfaces.tex_temperature = new std::shared_ptr <GLTexture>();
    *Surfaces.tex_temperature = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_R32F, GL_RED);

    kk.FboHandle = (*Surfaces.tex_temperature)->frame_buffer();
    kk.ColorTexture = (*Surfaces.tex_temperature)->handle();
    kk.Width = (*Surfaces.tex_temperature)->width();
    kk.Height = (*Surfaces.tex_temperature)->height();
    kk.Depth = (*Surfaces.tex_temperature)->depth();

    Surfaces.temperature_ = kk;

    gb1 = new std::shared_ptr <GLTexture>();
    *gb1 = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_R32F, GL_RED);

    kk.FboHandle = (*gb1)->frame_buffer();
    kk.ColorTexture = (*gb1)->handle();
    kk.Width = (*gb1)->width();
    kk.Height = (*gb1)->height();
    kk.Depth = (*gb1)->depth();

    general_buffers.general_buffer_1 = kk;

    gb3 = new std::shared_ptr <GLTexture>();
    *gb3 = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_RGBA32F, GL_RGBA);

    kk.FboHandle = (*gb3)->frame_buffer();
    kk.ColorTexture = (*gb3)->handle();
    kk.Width = (*gb3)->width();
    kk.Height = (*gb3)->height();
    kk.Depth = (*gb3)->depth();

    general_buffers.general_buffer_3 = kk;
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
        float sin_factor = static_cast<float>(sin(0 / 4 * Pi));
        float cos_factor = static_cast<float>(cos(0 / 4 * Pi));
        float hotspot_x =
            cos_factor * SplatRadius * 0.8f + kImpulsePosition.getX();
        float hotspot_z =
            sin_factor * SplatRadius * 0.8f + kImpulsePosition.getZ();
        Vector3 hotspot(hotspot_x, 0, hotspot_z);

        glBindBuffer(GL_ARRAY_BUFFER, Vbos.FullscreenQuad);
        glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);
        glViewport(0, 0, GridWidth, GridHeight);

        Metrics::Instance()->OnFrameUpdateBegins();

        // TODO
        MultigridPoissonSolver sss;
        //sss.Diagnose(Surfaces.tex_velocity->get());
        
        // Advect velocity
        //CudaMain::Instance()->AdvectVelocity(*Surfaces.tex_velocity, *gb3, delta_time, VelocityDissipation);
        Advect(Surfaces.velocity_, Surfaces.velocity_, SurfacePod(), general_buffers.general_buffer_3, delta_time, VelocityDissipation);
        std::swap(*Surfaces.tex_velocity, *gb3);
        std::swap(Surfaces.velocity_, general_buffers.general_buffer_3);

        //sss.Diagnose(Surfaces.tex_velocity->get());

        Metrics::Instance()->OnVelocityAvected();

        // Advect density and temperature
        ClearSurface(gb1->get(), 0.0f);
        //CudaMain::Instance()->Advect(*Surfaces.tex_velocity, *Surfaces.tex_temperature, *gb1, delta_time, TemperatureDissipation);
        Advect(Surfaces.velocity_, Surfaces.temperature_, SurfacePod(), general_buffers.general_buffer_1, delta_time, TemperatureDissipation);
        std::swap(*Surfaces.tex_temperature, *gb1);
        std::swap(Surfaces.temperature_, general_buffers.general_buffer_1);
        Metrics::Instance()->OnTemperatureAvected();

        ClearSurface(gb1->get(), 0.0f);
        //CudaMain::Instance()->Advect(*Surfaces.tex_velocity, *Surfaces.tex_density, *gb1, delta_time, DensityDissipation);
        Advect(Surfaces.velocity_, Surfaces.density_, SurfacePod(), general_buffers.general_buffer_1, delta_time, DensityDissipation);
        std::swap(*Surfaces.tex_density, *gb1);
        std::swap(Surfaces.density_, general_buffers.general_buffer_1);
        Metrics::Instance()->OnDensityAvected();
        
        // Apply buoyancy and gravity
        //CudaMain::Instance()->ApplyBuoyancy(*Surfaces.tex_velocity, *Surfaces.tex_temperature, *gb3, delta_time, AmbientTemperature, kBuoyancyCoef, SmokeWeight);
        ApplyBuoyancy(Surfaces.velocity_, Surfaces.temperature_, general_buffers.general_buffer_3, delta_time);
        std::swap(*Surfaces.tex_velocity, *gb3);
        std::swap(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnBuoyancyApplied();

        // Splat new smoke
        CudaMain::Instance()->ApplyImpulse(*Surfaces.tex_density, kImpulsePosition, hotspot, SplatRadius, ImpulseDensity);
        CudaMain::Instance()->ApplyImpulse(*Surfaces.tex_temperature, kImpulsePosition, hotspot, SplatRadius, ImpulseTemperature);
        //ApplyImpulse(Surfaces.density_, kImpulsePosition, hotspot, ImpulseDensity, ImpulseDensity);
        //ApplyImpulse(Surfaces.temperature_, kImpulsePosition, hotspot, ImpulseTemperature, ImpulseTemperature);
        Metrics::Instance()->OnImpulseApplied();

        // TODO: Try to slightly optimize the calculation by pre-multiplying 1/h^2.
        ComputeDivergence(Surfaces.velocity_, SurfacePod(),
                          general_buffers.general_buffer_3);
        Metrics::Instance()->OnDivergenceComputed();

        // Solve pressure-velocity Poisson equation
        SolvePressure(general_buffers.general_buffer_3, *gb3);
        Metrics::Instance()->OnPressureSolved();

        // Rectify velocity via the gradient of pressure
        SubtractGradient(Surfaces.velocity_, general_buffers.general_buffer_3);
        Metrics::Instance()->OnVelocityRectified();

        CudaMain::Instance()->RoundPassed(frame_count);
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
    ClearSurface(Surfaces.tex_velocity->get(), 0.0f);
    ClearSurface(Surfaces.tex_density->get(), 0.0f);
    ClearSurface(Surfaces.tex_temperature->get(), 0.0f);
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
