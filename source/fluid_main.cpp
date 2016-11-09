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
GLuint point_vbo_ = 0;
int kNumParticles = 4000000;
GLProgram blob_program_;


#define STRINGIFY(A) #A

const char* blob_vs = STRINGIFY(

in vec4 in_position;

void main()
{
    gl_Position = in_position;
}
);

const char* blob_gs = STRINGIFY(

layout(points)           in;
layout(triangle_strip)   out;
layout(max_vertices = 4) out;

uniform mat4 u_mv_matrix;
uniform mat4 u_mvp_matrix;
uniform vec3 u_hori;
uniform vec3 u_vert;

out vec2 gs_tex_coord;

void main()
{
    vec4 pos = u_mvp_matrix * gl_in[0].gl_Position;

    vec3 pos_in_eye_coord = vec3(u_mv_matrix * gl_in[0].gl_Position);
    float dist = length(pos_in_eye_coord);
    float blob_size = 250.0f / dist;

    gl_Position = pos + vec4(-blob_size, -blob_size, 0, 0);
    gs_tex_coord = vec2(0.0f, 0.0f);
    EmitVertex();

    gl_Position = pos + vec4(-blob_size, blob_size, 0, 0);
    gs_tex_coord = vec2(0.0f, 1.0f);
    EmitVertex();

    gl_Position = pos + vec4(blob_size, -blob_size, 0, 0);
    gs_tex_coord = vec2(1.0f, 0.0f);
    EmitVertex();

    gl_Position = pos + vec4(blob_size, blob_size, 0, 0);
    gs_tex_coord = vec2(1.0f, 1.0f);
    EmitVertex();

    EndPrimitive();
}
);

const char *blob_fs = STRINGIFY(

uniform mat4 u_mv_matrix;
uniform vec3 u_light_dir;

in vec2 gs_tex_coord;
out vec4 out_color;

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    const vec3 light_dir = vec3(0.1, 0.75, 1.0);

    vec3 blob_normal;
    blob_normal.xy = gs_tex_coord.xy * 2.0f - 1.0f;
    float mag = dot(blob_normal.xy, blob_normal.xy);
    if (mag > 1.0f)
        discard;

    blob_normal.z = sqrt(1.0f - mag);

    float diffuse = max(0.0f, dot(light_dir, blob_normal));

    vec3 color1 = vec3(63.0f, 194.0f, 250.0f);
    vec3 color2 = vec3(25.0f, 88.0f, 133.0f);

    vec3 color3 = vec3(80.0f, 228.0f, 255.0f); // 126,240,158
    vec3 color4 = vec3(63.0f, 178.0f, 239.0f); // 134,203,142
    vec3 color5 = vec3(49.0f, 140.0f, 188.0f); // 134,141,112
    vec3 hsv_color = rgb2hsv(color3);

//     float alpha = (diffuse);
//     vec3 color = color1 * alpha + color2 * (1.0f - alpha);
    float hue_factor = 1.0f + (1.0f - diffuse * diffuse) * 0.0635f;
    hsv_color.x *= min(hue_factor, 1.0635f);
    hsv_color.y *= diffuse * 0.4f + 0.6f;
    hsv_color.z *= diffuse * 0.3f + 0.7f;

    out_color = vec4(hsv2rgb(hsv_color) / 255.0f, 1.0f);
}
);


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

    glm::vec3 eye(0.0f, 0.0f, 150.0f + trackball_->GetZoom());
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    float aspect_radio =
        static_cast<float>(viewport_size_.x) / viewport_size_.y;
//     renderer_->Update(
//         eye, glm::lookAt(eye, target, up), glm::mat4(trackball_->GetRotation()),
//         glm::perspective(kFieldOfView_, aspect_radio, 0.0f, 1.0f));
    persp = glm::perspective(kFieldOfView_, aspect_radio, -1.0f, 1.0f);
    mv = glm::lookAt(eye, target, up) *
        glm::mat4(trackball_->GetRotation()) * translate;

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

    //glClearColor(0.01f, 0.06f, 0.08f, 0.0f);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    //glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    //glClearColor(0.6f, 0.6f, 0.6f, 0.6f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    float focal_length = 1.0f / std::tan(kFieldOfView_ / 2);
    //renderer_->Render(sim_->GetDensityField(), focal_length);

    Metrics::Instance()->OnRaycastPerformed();
    Metrics::Instance()->OnFrameRendered();

    DisplayMetrics();
    //return;

    // TODO ====================================================================
    
    CudaMain::Instance()->CopyToVbo(point_vbo_, &g_cmfp);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);

    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);

    blob_program_.Use();
    blob_program_.SetUniform("u_mv_matrix", mv);
    blob_program_.SetUniform("u_mvp_matrix", persp * mv);
    blob_program_.SetUniform("u_light_dir",
                             FluidConfig::Instance()->light_position());

    glBindBuffer(GL_ARRAY_BUFFER, point_vbo_);
    glVertexPointer(3, GL_HALF_FLOAT, 0, 0);
    glVertexAttribPointer(SlotPosition, 3, GL_HALF_FLOAT, GL_FALSE,
                          0, nullptr);
    glEnableVertexAttribArray(SlotPosition);
    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawArrays(GL_POINTS, 0, kNumParticles);
    GLenum e = glGetError();
    assert(e == GL_NO_ERROR);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    blob_program_.Unuse();
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_DEPTH_TEST);
    // =========================================================================
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

    renderer_->OnViewportSized(viewport_size_);
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

    // TODO ====================================================================
    
    point_vbo_ = CreateDynamicVbo(kNumParticles);
    CudaMain::Instance()->RegisterGLBuffer(point_vbo_);
    blob_program_.Load(blob_vs, blob_gs, blob_fs);
    // =========================================================================

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
