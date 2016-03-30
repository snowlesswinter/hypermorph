#include "stdafx.h"

#include <sstream>

#include "cuda_host/cuda_main.h"
#include "fluid_simulator.h"
#include "graphics_volume.h"
#include "metrics.h"
#include "opengl/gl_program.h"
#include "opengl/gl_texture.h"
#include "overlay_content.h"
#include "shader/fluid_shader.h"
#include "shader/raycast_shader.h"
#include "third_party/opengl/freeglut.h"
#include "third_party/opengl/glew.h"
#include "utility.h"

int timer_interval = 10; // ms
int main_frame_handle = 0;
ITrackball* track_ball;
vmath::Point3 EyePosition;
GLuint RaycastProgram;
float FieldOfView = 0.7f;
bool SimulateFluid = true;
OverlayContent overlay_;
GLProgram advect_packed_program_;
int kMainWindowWidth = ViewportWidth;
int kMainWindowHeight = ViewportWidth;
LARGE_INTEGER time_freq;
LARGE_INTEGER prev_time;
int g_diagnosis = 0;
FluidSimulator* sim_ = nullptr;

struct
{
    vmath::Matrix4 Projection;
    vmath::Matrix4 Modelview;
    vmath::Matrix4 View;
    vmath::Matrix4 ModelviewProjection;
} Matrices;

struct
{
    GLuint CubeCenter;
    GLuint FullscreenQuad;
} Vbos;

void Cleanup(int exit_code)
{
    if (main_frame_handle)
        glutDestroyWindow(main_frame_handle);

    CudaMain::DestroyInstance();
    exit(EXIT_SUCCESS);
}

bool InitGraphics(int* argc, char** argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(kMainWindowWidth, kMainWindowHeight);
    glutInitWindowPosition(
        (glutGet(GLUT_SCREEN_WIDTH) - kMainWindowWidth) / 2,
        (glutGet(GLUT_SCREEN_HEIGHT) - kMainWindowHeight) / 2);
    main_frame_handle = glutCreateWindow("Fluid Simulation");

    // initialize necessary OpenGL extensions
    glewInit();

    if (!glewIsSupported(
        "GL_VERSION_2_0 "
        "GL_ARB_pixel_buffer_object "
        "GL_EXT_framebuffer_object "
        ))
    {
        PrintDebugString("ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, kMainWindowWidth, kMainWindowHeight);
    PrintDebugString("OpenGL Version: %s\n", glGetString(GL_VERSION));

    CudaMain::Instance();
    return true;
}

void UpdateFrame(unsigned int microseconds)
{
    float dt = microseconds * 0.000001f;
    track_ball->Update(microseconds);
    EyePosition = vmath::Point3(0, 0, 3.5f + track_ball->GetZoom());
    vmath::Vector3 up(0, 1, 0); vmath::Point3 target(0);
    Matrices.View = vmath::Matrix4::lookAt(EyePosition, target, up);
    vmath::Matrix4 modelMatrix(transpose(track_ball->GetRotation()), vmath::Vector3(0));
    modelMatrix *= vmath::Matrix4::rotationY(0.5f);
    Matrices.Modelview = Matrices.View * modelMatrix;

    Matrices.Projection = vmath::Matrix4::perspective(
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

    if (SimulateFluid)
    {
        glBindBuffer(GL_ARRAY_BUFFER, Vbos.FullscreenQuad);
        glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);
        glViewport(0, 0, GridWidth, GridHeight);
        sim_->Update(delta_time, time_elapsed, frame_count);
    }
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
    for (int i = 0; i < sizeof(o) / sizeof(o[0]); i++)
    {
        float cost = Metrics::Instance()->GetOperationTimeCost(
            static_cast<Metrics::Operations>(i));
        if (cost > 0.01f)
            text << o[i] << ": " << cost << std::endl;
    }

    overlay_.RenderText(text.str());
}

void RenderFrame()
{
    Metrics::Instance()->OnFrameRenderingBegins();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, ViewportWidth, ViewportHeight);
    glClearColor(0.01f, 0.06f, 0.08f, 0.0f);
    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBindBuffer(GL_ARRAY_BUFFER, Vbos.CubeCenter);
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glBindTexture(GL_TEXTURE_3D,
                  sim_->GetDensityTexture().gl_texture()->handle());
    glUseProgram(RaycastProgram);
    SetUniform("ModelviewProjection", Matrices.ModelviewProjection);
    SetUniform("Modelview", Matrices.Modelview);
    SetUniform("ViewMatrix", Matrices.View);
    SetUniform("ProjectionMatrix", Matrices.Projection);
    SetUniform("RayStartPoints", 1);
    SetUniform("RayStopPoints", 2);
    SetUniform("EyePosition", EyePosition);
    SetUniform("RayOrigin", vmath::Vector4(transpose(Matrices.Modelview) * EyePosition).getXYZ());
    SetUniform("FocalLength", 1.0f / std::tan(FieldOfView / 2));
    SetUniform("WindowSize", float(kMainWindowWidth), float(kMainWindowHeight));
    glDrawArrays(GL_POINTS, 0, 1);

    Metrics::Instance()->OnFrameRendered();

    DisplayMetrics();
}

void Display()
{
    LARGE_INTEGER currentTime;
    __int64 elapsed;
    double deltaTime;

    QueryPerformanceCounter(&currentTime);
    elapsed = currentTime.QuadPart - prev_time.QuadPart;
    deltaTime = elapsed * 1000000.0 / time_freq.QuadPart;
    prev_time = currentTime;

    UpdateFrame((unsigned int)deltaTime);
    RenderFrame();
    glutSwapBuffers();
}

void Keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case VK_ESCAPE:
            Cleanup(EXIT_SUCCESS);
            break;
        case VK_SPACE:
            SimulateFluid = !SimulateFluid;
            break;
        case 'd':
            Metrics::Instance()->set_diagnosis_mode(
                !Metrics::Instance()->diagnosis_mode());
            break;
        case 'g':
            g_diagnosis = 1 - g_diagnosis;
            sim_->set_diagnosis(!!g_diagnosis);
            break;
        case 'r':
            sim_->Reset();
            break;
        case '`':
            track_ball->ReturnHome();
            break;
    }
}

void Mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        track_ball->MouseDown(x, y);
    else if (state == GLUT_UP)
        track_ball->MouseUp(x, y);
}

void Wheel(int button, int state, int x, int y)
{
    float d = float(state * 60) / 1000 *
        std::max(abs(track_ball->GetZoom()), 1.0f);
    track_ball->MouseWheel(x, y, -d);
}

void Motion(int x, int y)
{
    track_ball->MouseMove(x, y);
}

void TimerProc(int value)
{
    glutPostRedisplay();
    glutTimerFunc(timer_interval, TimerProc, 0);
}

bool Initialize()
{
    sim_ = new FluidSimulator();
    sim_->set_graphics_lib(GRAPHICS_LIB_CUDA);
    if (!sim_->Init())
        return false;

    track_ball = CreateTrackball(ViewportWidth * 1.0f, ViewportHeight * 1.0f,
                                 ViewportWidth * 0.5f);
    RaycastProgram = LoadProgram(RaycastShader::Vertex(),
                                 RaycastShader::Geometry(),
                                 RaycastShader::Fragment());
    advect_packed_program_.Load(FluidShader::Vertex(), FluidShader::PickLayer(),
                                FluidShader::GetAdvectPackedShaderCode());
    Vbos.CubeCenter = CreatePointVbo(0, 0, 0);
    Vbos.FullscreenQuad = CreateQuadVbo();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableVertexAttribArray(SlotPosition);

    Metrics::Instance()->SetOperationSync([]() { glFinish(); });
    Metrics::Instance()->SetTimeSource(
        []() -> double { return GetCurrentTimeInSeconds(); });

    return true;
}

int __stdcall WinMain(HINSTANCE hInst, HINSTANCE ignoreMe0, LPSTR ignoreMe1, INT ignoreMe2)
{
    char* command_line = GetCommandLineA();
    int agrc = 1;
    if (!InitGraphics(&agrc, &command_line))
        return -1;

    if (!Initialize())
        Cleanup(EXIT_FAILURE);

    // register callbacks
    glutDisplayFunc(Display);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(Motion);
    glutMouseWheelFunc(Wheel);
    glutTimerFunc(timer_interval, TimerProc, 0);

    QueryPerformanceFrequency(&time_freq);
    QueryPerformanceCounter(&prev_time);

    glutMainLoop();

    Cleanup(EXIT_SUCCESS);
    return 0;
}
