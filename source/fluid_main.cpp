#include "stdafx.h"

#include <sstream>
#include <numeric>

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
#include "third_party/glm/gtc/matrix_transform.hpp"
#include "third_party/opengl/freeglut.h"
#include "third_party/opengl/glew.h"
#include "trackball.h"
#include "utility.h"
#include "volume_renderer.h"

int timer_interval_ = 10; // ms
int main_frame_handle_ = 0;
Trackball* trackball_ = nullptr;
float kFieldOfView_ = 0.7f;
float kAspectRatio = 1.0f;
bool simulate_fluid_ = true;
OverlayContent overlay_;
LARGE_INTEGER time_freq_;
LARGE_INTEGER prev_time_;
int g_diagnosis = 0;
FluidSimulator* sim_ = nullptr;
VolumeRenderer* renderer_ = nullptr;
ConfigFileWatcher* watcher_ = nullptr;
glm::ivec2 viewport_size_(0);

struct
{
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

    if (main_frame_handle_)
        glutDestroyWindow(main_frame_handle_);

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
    glutInitWindowSize(viewport_size_.x, viewport_size_.y);
    glutInitWindowPosition(
        (glutGet(GLUT_SCREEN_WIDTH) - viewport_size_.x) / 2,
        (glutGet(GLUT_SCREEN_HEIGHT) - viewport_size_.y) / 2);
    main_frame_handle_ = glutCreateWindow("Fluid Simulation");

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
    glViewport(0, 0, viewport_size_.x, viewport_size_.y);
    PrintDebugString("OpenGL Version: %s\n", glGetString(GL_VERSION));

    CudaMain::Instance();
    return true;
}

void UpdateFrame(unsigned int microseconds)
{
    float delta_time = microseconds * 0.000001f;
    trackball_->Update(microseconds);

    glm::vec3 eye(0.0f, 0.0f, 3.8f + trackball_->GetZoom());
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    renderer_->Update(
        eye, glm::lookAt(eye, target, up), glm::mat4(trackball_->GetRotation()),
        glm::perspective(kFieldOfView_, kAspectRatio, 0.0f, 1.0f));

    static double time_elapsed = 0;
    time_elapsed += delta_time;

    static int frame_count = 0;
    frame_count++;

    if (simulate_fluid_) {
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
        "Vorticity",
        "Raycast",
        "Render",
        "Prolongate",
    };
    for (int i = 0; i < sizeof(o) / sizeof(o[0]); i++) {
        float cost = Metrics::Instance()->GetOperationTimeCost(
            static_cast<Metrics::Operations>(i));
        if (cost > 0.01f)
            text << o[i] << ": " << cost << std::endl;
    }

    overlay_.RenderText(text.str(), viewport_size_.x, viewport_size_.y);
}

void RenderFrame()
{
    Metrics::Instance()->OnFrameRenderingBegins();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClearColor(0.01f, 0.06f, 0.08f, 0.0f);
    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    //glClearColor(0.6f, 0.6f, 0.6f, 0.6f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    float focal_length = 1.0f / std::tan(kFieldOfView_ / 2);
    renderer_->Render(sim_->GetDensityField(), focal_length);
    Metrics::Instance()->OnRaycastPerformed();
    Metrics::Instance()->OnFrameRendered();

    DisplayMetrics();
}

bool ResetRenderer()
{
    if (renderer_)
        delete renderer_;

    renderer_ = new VolumeRenderer();
    renderer_->set_graphics_lib(FluidConfig::Instance()->graphics_lib());
    return renderer_->Init(viewport_size_);
}

bool ResetSimulator()
{
    if (sim_)
        delete sim_;

    sim_ = new FluidSimulator();
    sim_->set_graphics_lib(FluidConfig::Instance()->graphics_lib());
    sim_->set_solver_choice(FluidConfig::Instance()->poisson_method());
    sim_->set_diagnosis(!!g_diagnosis);

    bool r = sim_->Init();
    if (r)
        sim_->NotifyConfigChanged();

    return r;
}

void Display()
{
    LARGE_INTEGER currentTime;
    __int64 elapsed;
    double deltaTime;

    QueryPerformanceCounter(&currentTime);
    elapsed = currentTime.QuadPart - prev_time_.QuadPart;
    deltaTime = elapsed * 1000000.0 / time_freq_.QuadPart;
    prev_time_ = currentTime;

    if (watcher_->file_modified()) {
        FluidConfig::Instance()->Reload();
        watcher_->ResetState();

        sim_->NotifyConfigChanged();
    }

    UpdateFrame(static_cast<unsigned int>(deltaTime));
    RenderFrame();
    glutSwapBuffers();
}

void Reshape(int w, int h)
{
    viewport_size_ = glm::ivec2(w, h);

    renderer_->OnViewportSized(viewport_size_);
    trackball_->OnViewportSized(viewport_size_);
}

void Keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case VK_ESCAPE:
            Cleanup(EXIT_SUCCESS);
            break;
        case VK_SPACE:
            simulate_fluid_ = !simulate_fluid_;
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
            ResetRenderer();
            Metrics::Instance()->Reset();
            break;
        case '`':
            trackball_->ReturnHome();
            break;
    }
}

bool CalculateImpulseSpot(int x, int y, glm::vec2* result)
{
    int width = glm::min(viewport_size_.x, viewport_size_.y);
    if (x < (viewport_size_.x - width) / 2 ||
            x >= (viewport_size_.x + width) / 2 ||
            y < (viewport_size_.y - width) / 2 ||
            y >= (viewport_size_.y + width) / 2)
        return false;

    glm::vec3 eye(0.0f, 0.0f, 3.8f + trackball_->GetZoom());
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    glm::mat4 view = glm::lookAt(eye, target, up);
    glm::mat4 model_view = view * glm::mat4(trackball_->GetRotation());
    glm::vec3 eye_position =
        (glm::transpose(model_view) * glm::vec4(eye, 1.0f)).xyz();
    glm::vec3 screen_hit(2.0f * x / viewport_size_.x - 1.0f,
                         1.0f - 2.0f * y / viewport_size_.y,
                         1.0f);
    screen_hit =
        (glm::transpose(model_view) * glm::vec4(screen_hit, 1.0f)).xyz();

    glm::vec3 diff = screen_hit - eye_position;
    if (glm::abs(diff.y) < std::numeric_limits<float>::epsilon()) {
        return false;
    }

    float hot_plane_y = -1.0f + (2.0f / GridHeight) * 2.0f;
    float r = (hot_plane_y - eye_position.y) / diff.y;
    result->x = r * diff.x + eye_position.x;
    result->y = r * diff.z + eye_position.z;
    return true;
}

void Mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        if (glutGetModifiers() == GLUT_ACTIVE_CTRL) {
            glm::vec2 hotspot;
            if (CalculateImpulseSpot(x, y, &hotspot))
                sim_->StartImpulsing(hotspot.x, hotspot.y);
        } else {
            trackball_->MouseDown(x, y);
        }
    } else if (state == GLUT_UP) {
        trackball_->MouseUp(x, y);
        sim_->StopImpulsing();
    }
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

    glm::vec2 hotspot;
    if (CalculateImpulseSpot(x, y, &hotspot))
        sim_->UpdateImpulsing(hotspot.x, hotspot.y);
}

void TimerProc(int value)
{
    glutPostRedisplay();
    glutTimerFunc(timer_interval_, TimerProc, 0);
}

bool Initialize()
{
    if (!ResetSimulator())
        return false;

    if (!ResetRenderer())
        return false;

    int radius = std::min(viewport_size_.x, viewport_size_.y);
    trackball_ = Trackball::CreateTrackball(viewport_size_.x, viewport_size_.y,
                                            radius * 0.5f);

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
    viewport_size_.x = FluidConfig::Instance()->initial_viewport_width();
    viewport_size_.y = FluidConfig::Instance()->initial_viewport_height();

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
    glutTimerFunc(timer_interval_, TimerProc, 0);

    QueryPerformanceFrequency(&time_freq_);
    QueryPerformanceCounter(&prev_time_);

    glutMainLoop();

    Cleanup(EXIT_SUCCESS);
    return 0;
}
