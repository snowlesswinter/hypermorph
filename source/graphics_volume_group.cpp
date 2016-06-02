#include "stdafx.h"
#include "graphics_volume_group.h"

#include <cassert>

GraphicsVolume3::GraphicsVolume3(GraphicsLib graphics_lib)
    : graphics_lib_(graphics_lib)
    , x_()
    , y_()
    , z_()
    , v_({&x_, &y_, &z_})
{
}

GraphicsVolume3::GraphicsVolume3(GraphicsVolume3&& obj)
    : graphics_lib_(obj.graphics_lib_)
    , x_(std::move(obj.x_))
    , y_(std::move(obj.y_))
    , z_(std::move(obj.z_))
    , v_({&x_, &y_, &z_})
{
}

GraphicsVolume3::GraphicsVolume3(const std::shared_ptr<GraphicsVolume>& x,
                                 const std::shared_ptr<GraphicsVolume>& y,
                                 const std::shared_ptr<GraphicsVolume>& z)
    : graphics_lib_(x->graphics_lib())
    , x_()
    , y_()
    , z_()
    , v_({&x_, &y_, &z_})
{
    Assign(x, y, z);
}

bool GraphicsVolume3::Create(int width, int height, int depth,
                             int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> x =
        std::make_shared<GraphicsVolume>(graphics_lib_);
    bool r = x->Create(width, height, depth, num_of_components, byte_width);
    assert(r);
    if (!r)
        return false;

    std::shared_ptr<GraphicsVolume> y =
        std::make_shared<GraphicsVolume>(graphics_lib_);
    r = y->Create(width, height, depth, num_of_components, byte_width);
    assert(r);
    if (!r)
        return false;

    std::shared_ptr<GraphicsVolume> z =
        std::make_shared<GraphicsVolume>(graphics_lib_);
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

bool GraphicsVolume3::Assign(const std::shared_ptr<GraphicsVolume>& x,
                             const std::shared_ptr<GraphicsVolume>& y,
                             const std::shared_ptr<GraphicsVolume>& z)
{
    if (!x)
        return false;

    if (!y || !y->HasSameProperties(*x))
        return false;

    if (!z || !z->HasSameProperties(*y))
        return false;

    x_ = x;
    y_ = y;
    z_ = z;
    return true;
}
