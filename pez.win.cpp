// Pez was developed by Philip Rideout and released under the MIT License.

#define _WIN32_WINNT 0x0500
#define WINVER 0x0500

#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <pez.h>

#include "opengl/glew.h"
#include "opengl/freeglut.h"

int timer_interval = 10; // ms
int main_frame_handle = 0;

void Cleanup(int exit_code)
{
    if (main_frame_handle)
        glutDestroyWindow(main_frame_handle);

    exit(EXIT_SUCCESS);
}

bool InitGL(int* argc, char** argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(PezGetConfig().Width, PezGetConfig().Height);
    main_frame_handle = glutCreateWindow("Fluid Simulation");

    // initialize necessary OpenGL extensions
    glewInit();

    if (!glewIsSupported(
        "GL_VERSION_2_0 "
        "GL_ARB_pixel_buffer_object "
        "GL_EXT_framebuffer_object "
        ))
    {
        PezDebugString("ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, PezGetConfig().Width, PezGetConfig().Height);
    PezDebugString("OpenGL Version: %s\n", glGetString(GL_VERSION));

    return true;
}

LARGE_INTEGER time_freq;
LARGE_INTEGER prev_time;
void Display(void)
{
    LARGE_INTEGER currentTime;
    __int64 elapsed;
    double deltaTime;

    QueryPerformanceCounter(&currentTime);
    elapsed = currentTime.QuadPart - prev_time.QuadPart;
    deltaTime = elapsed * 1000000.0 / time_freq.QuadPart;
    prev_time = currentTime;

    PezUpdate((unsigned int)deltaTime);
    PezRender();
    glutSwapBuffers();
}

void Keyboard(unsigned char key, int x, int y)
{
    PezHandleKey(key);
    switch (key)
    {
        case VK_ESCAPE:
            PostQuitMessage(0);
            break;
        case VK_OEM_2: // Question Mark / Forward Slash for US Keyboards
            break;
    }
}

void Mouse(int button, int state, int x, int y)
{
    int action = state == GLUT_DOWN ? PEZ_DOWN : PEZ_UP;
    switch (button)
    {
        case GLUT_LEFT_BUTTON:
            PezHandleMouse(x, y, action | PEZ_LEFT, 0);
            break;

        case GLUT_RIGHT_BUTTON:
            PezHandleMouse(x, y, action | PEZ_RIGHT, 0);
            break;

        case 4:
        {
            //
            break;
        }
    }
}

void Wheel(int button, int state, int x, int y)
{
    PezHandleMouse(x, y, PEZ_WHEEL, state * 60);
}

void Motion(int x, int y)
{
    PezHandleMouse(x, y, PEZ_MOVE | PEZ_LEFT, 0);
}

void TimerProc(int value)
{
    glutPostRedisplay();
    glutTimerFunc(timer_interval, TimerProc, 0);
}

int __stdcall WinMain(HINSTANCE hInst, HINSTANCE ignoreMe0, LPSTR ignoreMe1, INT ignoreMe2)
{
    char* command_line = GetCommandLineA();
    int agrc = 1;
    if (!InitGL(&agrc, &command_line))
        return -1;

    // register callbacks
    glutDisplayFunc(Display);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(Motion);
    glutMouseWheelFunc(Wheel);
    glutTimerFunc(timer_interval, TimerProc, 0);

    PezInitialize();
    QueryPerformanceFrequency(&time_freq);
    QueryPerformanceCounter(&prev_time);

    glutMainLoop();

    Cleanup(EXIT_SUCCESS);
    return 0;
}

void PezDebugString(const char* pStr, ...)
{
    char msg[1024] = {0};

    va_list a;
    va_start(a, pStr);

    _vsnprintf_s(msg, _countof(msg), _TRUNCATE, pStr, a);
    OutputDebugStringA(msg);
}

void _PezFatalError(const char* pStr, va_list a)
{
    char msg[1024] = {0};
    _vsnprintf_s(msg, _countof(msg), _TRUNCATE, pStr, a);
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
//    __debugbreak();
    exit(1);
}

void PezFatalError(const char* pStr, ...)
{
    va_list a;
    va_start(a, pStr);
    _PezFatalError(pStr, a);
}

void PezCheckCondition(int condition, ...)
{
    va_list a;
    const char* pStr;

    if (condition)
        return;

    va_start(a, condition);
    pStr = va_arg(a, const char*);
    _PezFatalError(pStr, a);
}