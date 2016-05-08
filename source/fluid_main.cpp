#include "stdafx.h"

#include <sstream>

#include "config_file_watcher.h"
#include "cuda_host/cuda_main.h"
#include "fluid_config.h"
#include "fluid_simulator.h"
#include "graphics_volume.h"
#include "metrics.h"
#include "opengl/gl_program.h"
#include "opengl/gl_volume.h"
#include "overlay_content.h"
#include "shader/fluid_shader.h"
#include "shader/raycast_shader.h"
#include "third_party/opengl/freeglut.h"
#include "third_party/opengl/glew.h"
#include "utility.h"
#include "volume_renderer.h"
#include "trackball.h"

int timer_interval = 10; // ms
int main_frame_handle = 0;
Trackball* trackball_ = nullptr;
vmath::Point3 EyePosition;
GLuint RaycastProgram;
float FieldOfView = 0.7f;
bool SimulateFluid = true;
OverlayContent overlay_;
LARGE_INTEGER time_freq;
LARGE_INTEGER prev_time;
int g_diagnosis = 0;
FluidSimulator* sim_ = nullptr;
VolumeRenderer* renderer_ = nullptr;
ConfigFileWatcher* watcher_ = nullptr;
int viewport_width_ = 0;
int viewport_height_ = 0;

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
    if (renderer_) {
        delete renderer_;
        renderer_ = nullptr;
    }

    if (watcher_) {
        delete watcher_;
        watcher_ = nullptr;
    }

    if (main_frame_handle)
        glutDestroyWindow(main_frame_handle);

    if (trackball_) {
        delete trackball_;
        trackball_ = nullptr;
    }

    CudaMain::DestroyInstance();
    exit(EXIT_SUCCESS);
}

bool InitGraphics(int* argc, char** argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(viewport_width_, viewport_height_);
    glutInitWindowPosition(
        (glutGet(GLUT_SCREEN_WIDTH) - viewport_width_) / 2,
        (glutGet(GLUT_SCREEN_HEIGHT) - viewport_height_) / 2);
    main_frame_handle = glutCreateWindow("Fluid Simulation");

    // initialize necessary OpenGL extensions
    glewInit();

    if (!glewIsSupported(
            "GL_VERSION_2_0 "
            "GL_ARB_pixel_buffer_object "
            "GL_EXT_framebuffer_object "
            )) {
        PrintDebugString("ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, viewport_width_, viewport_height_);
    PrintDebugString("OpenGL Version: %s\n", glGetString(GL_VERSION));

    CudaMain::Instance();
    return true;
}

void UpdateFrame(unsigned int microseconds)
{
    float delta_time = microseconds * 0.000001f;
    trackball_->Update(microseconds);
    EyePosition = vmath::Point3(0, 0, 3.8f + trackball_->GetZoom());
    vmath::Vector3 up(0, 1, 0);
    vmath::Point3 target(0);
    Matrices.View = vmath::Matrix4::lookAt(EyePosition, target, up);
    vmath::Matrix4 modelMatrix(trackball_->GetRotation(), vmath::Vector3(0));
    Matrices.Modelview = Matrices.View * modelMatrix;

    Matrices.Projection = vmath::Matrix4::perspective(
        FieldOfView,
        float(viewport_width_) / viewport_height_, // Aspect Ratio
        0.0f,   // Near Plane
        1.0f);  // Far Plane

    Matrices.ModelviewProjection = Matrices.Projection * Matrices.Modelview;
    static double time_elapsed = 0;
    time_elapsed += delta_time;

    static int frame_count = 0;
    frame_count++;

    if (SimulateFluid) {
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
    for (int i = 0; i < sizeof(o) / sizeof(o[0]); i++) {
        float cost = Metrics::Instance()->GetOperationTimeCost(
            static_cast<Metrics::Operations>(i));
        if (cost > 0.01f)
            text << o[i] << ": " << cost << std::endl;
    }

    overlay_.RenderText(text.str(), viewport_width_, viewport_height_);
}

void RenderFrame()
{
    Metrics::Instance()->OnFrameRenderingBegins();

    renderer_->Render(sim_->GetDensityTexture());

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewport_width_, viewport_height_);
    glClearColor(0.01f, 0.06f, 0.08f, 0.0f);
    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBindBuffer(GL_ARRAY_BUFFER, Vbos.CubeCenter);
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glBindTexture(GL_TEXTURE_3D,
                  sim_->GetDensityTexture()->gl_volume()->texture_handle());
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
    SetUniform("WindowSize", float(viewport_width_), float(viewport_height_));
    glDrawArrays(GL_POINTS, 0, 1);

    Metrics::Instance()->OnFrameRendered();

    DisplayMetrics();
}

bool ResetSimulator()
{
    if (sim_)
        delete sim_;

    sim_ = new FluidSimulator();
    sim_->set_graphics_lib(FluidConfig::Instance()->graphics_lib());
    sim_->set_solver_choice(FluidConfig::Instance()->poisson_method());
    sim_->set_num_multigrid_iterations(
        FluidConfig::Instance()->num_multigrid_iterations());
    sim_->set_num_full_multigrid_iterations(
        FluidConfig::Instance()->num_full_multigrid_iterations());
    sim_->set_diagnosis(!!g_diagnosis);
    return sim_->Init();
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

    if (watcher_->file_modified()) {
        FluidConfig::Instance()->Reload();
        watcher_->ResetState();
    }

    UpdateFrame(static_cast<unsigned int>(deltaTime));
    RenderFrame();
    glutSwapBuffers();
}

void Reshape(int w, int h)
{
    viewport_width_ = w;
    viewport_height_ = h;

    renderer_->OnViewportSized(w, h);
    trackball_->OnViewportSized(w, h);
}

void Keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case VK_ESCAPE:
            Cleanup(EXIT_SUCCESS);
            break;
        case VK_SPACE:
            SimulateFluid = !SimulateFluid;
            break;
        case 'd':
        case 'D':
            Metrics::Instance()->set_diagnosis_mode(
                !Metrics::Instance()->diagnosis_mode());
            break;
        case 'g':
        case 'G':
            g_diagnosis = 1 - g_diagnosis;
            sim_->set_diagnosis(!!g_diagnosis);
            break;
        case 'r':
        case 'R':
            FluidConfig::Instance()->Reload();
            ResetSimulator();
            Metrics::Instance()->Reset();
            break;
        case '`':
            trackball_->ReturnHome();
            break;
    }
}

void Mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        trackball_->MouseDown(x, y);
    else if (state == GLUT_UP)
        trackball_->MouseUp(x, y);
}

void Wheel(int button, int state, int x, int y)
{
    float d = float(state * 60) / 1000 *
        std::max(abs(trackball_->GetZoom()), 1.0f);
    trackball_->MouseWheel(x, y, -d);
}

void Motion(int x, int y)
{
    trackball_->MouseMove(x, y);
}

void TimerProc(int value)
{
    glutPostRedisplay();
    glutTimerFunc(timer_interval, TimerProc, 0);
}

bool Initialize()
{
    if (!ResetSimulator())
        return false;

    renderer_ = new VolumeRenderer();
    if (!renderer_->Init(viewport_width_, viewport_height_))
        return false;

    int radius = std::min(viewport_width_, viewport_height_);
    trackball_ = Trackball::CreateTrackball(viewport_width_, viewport_height_,
                                            radius * 0.5f);
    RaycastProgram = LoadProgram(RaycastShader::Vertex(),
                                 RaycastShader::Geometry(),
                                 RaycastShader::Fragment());
    Vbos.CubeCenter = CreatePointVbo(0, 0, 0);
    Vbos.FullscreenQuad = CreateQuadVbo();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableVertexAttribArray(SlotPosition);

    Metrics::Instance()->SetOperationSync(
        []() { glFinish(); CudaMain::Instance()->Sync(); });
    Metrics::Instance()->SetTimeSource(
        []() -> double { return GetCurrentTimeInSeconds(); });

    return true;
}

void LoadConfig()
{
    char file_path_buf[MAX_PATH] = {0};
    DWORD file_path_len = GetModuleFileNameA(nullptr, file_path_buf, MAX_PATH);
    std::string file_path = file_path_buf;
    file_path.replace(
        std::find(file_path.rbegin(), file_path.rend(), '\\').base(),
        file_path.end(), "fluid_config.txt");
    FluidConfig::Instance()->CreateIfNeeded(file_path);

    std::string cur_path(file_path);
    cur_path.erase(cur_path.find_last_of('\\'));
    std::string preset_path(cur_path);
    preset_path.erase(preset_path.find_last_of('\\'));
    preset_path.erase(preset_path.find_last_of('\\'));
    preset_path += "\\config";
    DWORD attrib = GetFileAttributesA(preset_path.c_str());

    if ((attrib != INVALID_FILE_ATTRIBUTES) &&
            (attrib & FILE_ATTRIBUTE_DIRECTORY)) {
        FluidConfig::Instance()->Load(file_path, preset_path);
    } else {
        FluidConfig::Instance()->Load(file_path, cur_path);
    }

    watcher_->StartWatching(cur_path);
}

int __stdcall WinMain(HINSTANCE inst, HINSTANCE ignore_me0, char* ignore_me1,
                      int ignore_me2)
{
    watcher_ = new ConfigFileWatcher();
    LoadConfig();

    char* command_line = GetCommandLineA();
    int argc = 1;
    viewport_width_ = FluidConfig::Instance()->initial_viewport_width();
    viewport_height_ = FluidConfig::Instance()->initial_viewport_height();

    if (!InitGraphics(&argc, &command_line))
        return -1;

    if (!Initialize())
        Cleanup(EXIT_FAILURE);

    // register callbacks
    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
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
