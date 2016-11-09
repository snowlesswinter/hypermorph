//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include "stdafx.h"

#include <sstream>
#include <numeric>

#include "blob_renderer.h"
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
float kFieldOfView_ = 1.0f;
bool simulate_fluid_ = true;
OverlayContent overlay_;
LARGE_INTEGER time_freq_;
LARGE_INTEGER prev_time_;
int g_diagnosis = 0;
FluidSimulator* sim_ = nullptr;
VolumeRenderer* volume_renderer_ = nullptr;
BlobRenderer* blob_renderer_ = nullptr;
ConfigFileWatcher* watcher_ = nullptr;
glm::ivec2 viewport_size_(0);


struct
{
    GLuint FullscreenQuad;
} Vbos;

void Cleanup(int exit_code)
{
    if (volume_renderer_) {
        delete volume_renderer_;
        volume_renderer_ = nullptr;
    }

    if (blob_renderer_) {
        delete blob_renderer_;
        blob_renderer_ = nullptr;
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

glm::ivec2 CalculateViewportSize()
{
    glm::vec3 grid_size = FluidConfig::Instance()->grid_size();
    float ref = std::min(grid_size.x, grid_size.y);

    glm::ivec2 result;
    result.x = FluidConfig::Instance()->initial_viewport_width();
    result.y = FluidConfig::Instance()->initial_viewport_width();

    result.x = static_cast<int>(result.x * grid_size.x / ref);
    result.y = static_cast<int>(result.y * grid_size.y / ref);

    float max_width = glutGet(GLUT_SCREEN_WIDTH) * 0.9f;
    if (result.x > max_width) {
        int x = result.x;
        result.x = static_cast<int>(max_width);

        result.y = static_cast<int>(result.y * max_width / x);
    }

    float max_height = glutGet(GLUT_SCREEN_HEIGHT) * 0.9f;
    if (result.y > max_height) {
        int y = result.y;
        result.y = static_cast<int>(max_height);

        result.x = static_cast<int>(result.x * max_height / y);
    }

    return result;
}

glm::ivec2 CalculateWindowPosition(const glm::ivec2& viewport_size)
{
    glm::ivec2 result;
    result.x = (glutGet(GLUT_SCREEN_WIDTH) - viewport_size.x) / 2;
    result.y = (glutGet(GLUT_SCREEN_HEIGHT) - 80 - viewport_size.y) / 2;
    return result;
}

bool InitGraphics(int* argc, char** argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);

    viewport_size_ = CalculateViewportSize();
    glutInitWindowSize(viewport_size_.x, viewport_size_.y);

    glm::ivec2 position = CalculateWindowPosition(viewport_size_);
    glutInitWindowPosition(position.x, position.y);
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

glm::mat4 mv;
glm::mat4 persp;


void UpdateFrame(unsigned int microseconds)
{
    float delta_time = microseconds * 0.000001f;
    trackball_->Update(microseconds);

    glm::vec3 half_size = FluidConfig::Instance()->grid_size() / 2.0f;
    glm::mat4 translate = glm::translate(
        glm::mat4(), glm::vec3(-half_size.x, -half_size.y, -half_size.z));

    float half_diag = sqrtf(half_size.x * half_size.x +
                            half_size.y * half_size.y +
                            half_size.z * half_size.z);
    float eye_dist = half_diag / sinf(kFieldOfView_ / 2);
    float near_pos = eye_dist - half_diag;
    float far_pos = eye_dist + half_diag;

    glm::vec3 eye(0.0f, 0.0f, eye_dist + trackball_->GetZoom());
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    float aspect_radio =
        static_cast<float>(viewport_size_.x) / viewport_size_.y;
//     volume_renderer_->Update(
//         eye, glm::lookAt(eye, target, up), glm::mat4(trackball_->GetRotation()),
//         glm::perspective(kFieldOfView_, aspect_radio, 0.0f, 1.0f));

    blob_renderer_->Update(
        eye, glm::mat4(trackball_->GetRotation()) * translate,
        glm::perspective(kFieldOfView_, aspect_radio, near_pos, far_pos));

    static double time_elapsed = 0;
    time_elapsed += delta_time;

    static int frame_count = 0;
    frame_count++;

    if (simulate_fluid_) {
        glBindBuffer(GL_ARRAY_BUFFER, Vbos.FullscreenQuad);
        glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);
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

        "FLIP Emission",
        "FLIP Interpolation",
        "FLIP Resampling",
        "FLIP Advection",
        "FLIP Cell Binding",
        "FLIP Prefix Sum",
        "FLIP Sorting",
        "FLIP Transfer",

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

    int n = Metrics::Instance()->GetActiveParticleNumber();
    if (n)
        text << "Active Particles: " << n << std::endl;

    overlay_.RenderText(text.str(), viewport_size_.x, viewport_size_.y);
}

CudaMain::FlipParticles g_cmfp;
void RenderFrame()
{
    Metrics::Instance()->OnFrameRenderingBegins();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    float focal_length = 1.0f / std::tan(kFieldOfView_ / 2);
    //volume_renderer_->Render(sim_->GetDensityField(), focal_length);
    blob_renderer_->Render();

    // FIXME:
    CudaMain::Instance()->CopyToVbo(blob_renderer_->point_vbo(), &g_cmfp);

    Metrics::Instance()->OnRaycastPerformed();
    Metrics::Instance()->OnFrameRendered();

    DisplayMetrics();
}

bool ResetRenderer()
{
    if (volume_renderer_)
        delete volume_renderer_;

    volume_renderer_ = new VolumeRenderer();
    volume_renderer_->set_graphics_lib(FluidConfig::Instance()->graphics_lib());
    bool result = volume_renderer_->Init(viewport_size_);

    if (blob_renderer_)
        delete blob_renderer_;

    blob_renderer_ = new BlobRenderer();
    blob_renderer_->set_graphics_lib(FluidConfig::Instance()->graphics_lib());
    blob_renderer_->set_fov(kFieldOfView_);
    result &= blob_renderer_->Init(FluidConfig::Instance()->max_num_particles(),
                                   viewport_size_);

    return result;
}

bool ResetSimulator()
{
    if (sim_)
        delete sim_;

    sim_ = new FluidSimulator();
    sim_->set_graphics_lib(FluidConfig::Instance()->graphics_lib());
    sim_->set_solver_choice(FluidConfig::Instance()->poisson_method());
    sim_->set_grid_size(FluidConfig::Instance()->grid_size());
    sim_->set_cell_size(FluidConfig::Instance()->cell_size());

    bool r = sim_->Init();
    if (r) {
        sim_->NotifyConfigChanged();
        sim_->set_diagnosis(g_diagnosis);
    }

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
    Sleep(1);
}

void Reshape(int w, int h)
{
    viewport_size_ = glm::ivec2(w, h);

    volume_renderer_->OnViewportSized(viewport_size_);
    blob_renderer_->OnViewportSized(viewport_size_);
    trackball_->OnViewportSized(viewport_size_);
}

void UpdateWindowPlacement()
{
    glm::ivec2 viewport_size = CalculateViewportSize();
    bool need_reposition = viewport_size != viewport_size_;
    glutReshapeWindow(viewport_size.x, viewport_size.y);

    if (need_reposition) {
        glm::ivec2 position = CalculateWindowPosition(viewport_size);
        glutPositionWindow(position.x, position.y);
    }
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
            g_diagnosis++;
            sim_->set_diagnosis(g_diagnosis);
            break;
        case 'h':
        case 'H':
            g_diagnosis = 0;
            sim_->set_diagnosis(g_diagnosis);
            break;
        case 'r':
        case 'R':
            UpdateWindowPlacement();
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

    float hot_plane_y = -1.0f + 0.015625f * 2.0f;
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
    if (!sim_->IsImpulsing())
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
