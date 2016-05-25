#include "stdafx.h"
#include "graphics_volume_group.h"

#include <cassert>

GraphicsVolume3::GraphicsVolume3()
    : x_()
    , y_()
    , z_()
    , v_({&x_, &y_, &z_})
{
}

GraphicsVolume3::GraphicsVolume3(GraphicsVolume3&& obj)
    : x_(std::move(obj.x_))
    , y_(std::move(obj.y_))
    , z_(std::move(obj.z_))
    , v_({&x_, &y_, &z_})
{
}

bool GraphicsVolume3::Create(int width, int height, int depth,
                             int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> x;
    bool r = x->Create(width, height, depth, num_of_components, byte_width);
    assert(r);
    if (!r)
        return false;

    std::shared_ptr<GraphicsVolume> y;
    r = y->Create(width, height, depth, num_of_components, byte_width);
    assert(r);
    if (!r)
        return false;

    std::shared_ptr<GraphicsVolume> z;
    r = z->Create(width, height, depth, num_of_components, byte_width);
    assert(r);
    if (!r)
        return false;

    x_ = x;
    y_ = y;
    z_ = z;
    return true;
}

const std::shared_ptr<GraphicsVolume>& GraphicsVolume3::operator[](int n) const
{
    assert(n < sizeof(v_) / sizeof(v_[0]));
    return *v_[n];
}

GraphicsVolume3::operator BoolType() const
{
    return (x_ && y_ && z_) ? &GraphicsVolume3::x_ : nullptr;
}