#include "stdafx.h"
#include "testing.h"

#include "third_party/opengl/glew.h"
#include "third_party/opengl/freeglut.h"
#include "fluid_unittest.h"
#include "multigrid_unittest.h"
#include "utility.h"

int APIENTRY wWinMain(HINSTANCE instance, HINSTANCE prev_instance,
                      wchar_t* command_line, int command_show)
{
    UNREFERENCED_PARAMETER(prev_instance);
    UNREFERENCED_PARAMETER(command_line);

    char* argv = GetCommandLineA();
    int argc = 1;
    glutInit(&argc, &argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);

    const int kTestWindowWidth = 512;
    glutInitWindowSize(kTestWindowWidth, kTestWindowWidth);
    glutInitWindowPosition(
        (glutGet(GLUT_SCREEN_WIDTH) - kTestWindowWidth) / 2,
        (glutGet(GLUT_SCREEN_HEIGHT) - kTestWindowWidth) / 2);
    int main_frame_handle = glutCreateWindow("Fluid Simulation Test");

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
    glViewport(0, 0, kTestWindowWidth, kTestWindowWidth);
    PrintDebugString("OpenGL Version: %s\n", glGetString(GL_VERSION));

    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableVertexAttribArray(SlotPosition);

    GLuint quad = CreateQuadVbo();

    glBindBuffer(GL_ARRAY_BUFFER, quad);
    glVertexAttribPointer(SlotPosition, 2, GL_SHORT, GL_FALSE, 2 * sizeof(short), 0);

    int random_seed = 0x56784321;
    //FluidUnittest::TestVelocityAdvection(random_seed);
    //FluidUnittest::TestDensityAdvection(random_seed);
    //FluidUnittest::TestTemperatureAdvection(random_seed);
    //FluidUnittest::TestBuoyancyApplication(random_seed);
    //FluidUnittest::TestDivergenceCalculation(random_seed);
    //FluidUnittest::TestDampedJacobi(random_seed);
    //FluidUnittest::TestGradientSubtraction(random_seed);

    MultigridUnittest::TestZeroGuessRelaxation(random_seed);
    MultigridUnittest::TestResidualCalculation(random_seed);
    MultigridUnittest::TestResidualRestriction(random_seed);
    MultigridUnittest::TestRestriction(random_seed);
    MultigridUnittest::TestProlongation(random_seed);

    if (main_frame_handle)
        glutDestroyWindow(main_frame_handle);

    return 0;
}