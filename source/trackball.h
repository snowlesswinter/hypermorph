#ifndef _TRACKBALL_H_
#define _TRACKBALL_H_

#include <stdint.h>

namespace Vectormath
{
namespace Aos
{
class Matrix3;
}
}

class Trackball
{
public:
    static Trackball* CreateTrackball(float width, float height, float radius);

    virtual void MouseDown(int x, int y) = 0;
    virtual void MouseUp(int x, int y) = 0;
    virtual void MouseMove(int x, int y) = 0;
    virtual void MouseWheel(int x, int y, float delta) = 0;
    virtual void ReturnHome() = 0;
    virtual Vectormath::Aos::Matrix3 GetRotation() const = 0;
    virtual float GetZoom() const = 0;
    virtual void Update(uint32_t microseconds) = 0;
};

#endif // _TRACKBALL_H_