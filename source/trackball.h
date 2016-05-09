#ifndef _TRACKBALL_H_
#define _TRACKBALL_H_

#include <stdint.h>

#include "third_party/glm/fwd.hpp"

class Trackball
{
public:
    static Trackball* CreateTrackball(int width, int height, float radius);

    virtual void MouseDown(int x, int y) = 0;
    virtual void MouseUp(int x, int y) = 0;
    virtual void MouseMove(int x, int y) = 0;
    virtual void MouseWheel(int x, int y, float delta) = 0;
    virtual void ReturnHome() = 0;
    virtual glm::mat3 GetRotation() const = 0;
    virtual float GetZoom() const = 0;
    virtual void Update(uint32_t microseconds) = 0;
    virtual void OnViewportSized(int width, int height) = 0;
};

#endif // _TRACKBALL_H_