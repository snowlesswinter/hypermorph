#include "Utility.h"
#include <cmath>

using namespace vmath;
using std::string;

static struct
{
    SurfacePod Velocity;
    SurfacePod density_;
    SurfacePod temperature_;
    SurfacePod Pressure;
} Surfaces;

static struct
{
    SurfacePod general_buffer_1;
    SurfacePod general_buffer_3;
} general_buffers;

static struct {
    Matrix4 Projection;
    Matrix4 Modelview;
    Matrix4 View;
    Matrix4 ModelviewProjection;
} Matrices;

static struct {
    GLuint CubeCenter;
    GLuint FullscreenQuad;
} Vbos;

static ITrackball* Trackball;
static Point3 EyePosition;
static GLuint RaycastProgram;
static float FieldOfView = 0.7f;
static bool SimulateFluid = true;

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

    Trackball = CreateTrackball(cfg.Width * 1.0f, cfg.Height * 1.0f, cfg.Width * 0.5f);
    RaycastProgram = LoadProgram("Raycast.VS", "Raycast.GS", "Raycast.FS");
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

void PezRender()
{
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
}

void PezUpdate(unsigned int microseconds)
{
    float dt = microseconds * 0.000001f;
    Trackball->Update(microseconds);
    EyePosition = Point3(0, 0, 3.5f + Trackball->GetZoom());
    Vector3 up(0, 1, 0); Point3 target(0);
    Matrices.View = Matrix4::lookAt(EyePosition, target, up);
    Matrix4 modelMatrix(transpose(Trackball->GetRotation()), Vector3(0));
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
    float delta_time = 0.33f;// dt * 10.0f;

    if (SimulateFluid) {
        double hotspot_x = cos(time_elapsed * Pi) * SplatRadius * 0.8 +
            impulse_position.getX();
        double hotspot_z = sin(time_elapsed * Pi) * SplatRadius * 0.8 +
            impulse_position.getZ();
        Vector3 hotspot(static_cast<float>(hotspot_x), 0,
                        static_cast<float>(hotspot_z));

        glBindBuffer(GL_ARRAY_BUFFER, Vbos.FullscreenQuad);
        glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);
        glViewport(0, 0, GridWidth, GridHeight);

        // Advect velocity
        Advect(Surfaces.Velocity, Surfaces.Velocity, SurfacePod(), general_buffers.general_buffer_3, delta_time, VelocityDissipation);
        std::swap(Surfaces.Velocity, general_buffers.general_buffer_3);

        // Advect temperature
        ClearSurface(general_buffers.general_buffer_1, 0);
        Advect(Surfaces.Velocity, Surfaces.temperature_, SurfacePod(), general_buffers.general_buffer_1, delta_time, TemperatureDissipation);
        std::swap(Surfaces.temperature_, general_buffers.general_buffer_1);

        // Advect density
        ClearSurface(general_buffers.general_buffer_1, 0);
        Advect(Surfaces.Velocity, Surfaces.density_, SurfacePod(), general_buffers.general_buffer_1, delta_time, DensityDissipation);
        std::swap(Surfaces.density_, general_buffers.general_buffer_1);

        // Apply buoyancy and gravity
        ApplyBuoyancy(Surfaces.Velocity, Surfaces.temperature_, Surfaces.density_, general_buffers.general_buffer_3, delta_time);
        std::swap(Surfaces.Velocity, general_buffers.general_buffer_3);

        // Splat new smoke
        ApplyImpulse(Surfaces.temperature_, impulse_position, hotspot, ImpulseTemperature);
        ApplyImpulse(Surfaces.density_, impulse_position, hotspot, ImpulseDensity);

        // Calculate divergence
        ClearSurface(general_buffers.general_buffer_1, 0);

        // TODO: Try to slightly optimize the calculation by pre-multiplying 1/h^2.
        ComputeDivergence(Surfaces.Velocity, SurfacePod(), general_buffers.general_buffer_1);

        // Solve pressure-velocity Poisson equation
        SolvePressure(Surfaces.Pressure, general_buffers.general_buffer_1, SurfacePod());

        // Rectify velocity via the gradient of pressure
        SubtractGradient(Surfaces.Velocity, Surfaces.Pressure, SurfacePod(), general_buffers.general_buffer_3);
        std::swap(Surfaces.Velocity, general_buffers.general_buffer_3);
    }
}

void PezHandleMouse(int x, int y, int action)
{
    if (action & PEZ_DOWN) Trackball->MouseDown(x, y);
    else if (action & PEZ_UP) Trackball->MouseUp(x, y);
    else if (action & PEZ_MOVE) Trackball->MouseMove(x, y);
    else if (action & PEZ_DOUBLECLICK) Trackball->ReturnHome();
}

void PezHandleKey(char c)
{
    SimulateFluid = !SimulateFluid;
}
