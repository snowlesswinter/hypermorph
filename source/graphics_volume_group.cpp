//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
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

bool GraphicsVolume3::Assign(const std::shared_ptr<GraphicsVolume>& x,
                             const std::shared_ptr<GraphicsVolume>& y,
                             const std::shared_ptr<GraphicsVolume>& z)
{
    if (!x)
        return false;

    assert(!y || y->HasSameProperties(*x));
    if (!y || !y->HasSameProperties(*x))
        return false;

    assert(!z || z->HasSameProperties(*y));
    if (!z || !z->HasSameProperties(*y))
        return false;

    x_ = x;
    y_ = y;
    z_ = z;
    return true;
}

bool GraphicsVolume3::Create(int width, int height, int depth,
                             int num_of_components, int byte_width, int border)
{
    std::shared_ptr<GraphicsVolume> x =
        std::make_shared<GraphicsVolume>(graphics_lib_);
    bool r = x->Create(width, height, depth, num_of_components, byte_width,
                       border);
    assert(r);
    if (!r)
        return false;

    std::shared_ptr<GraphicsVolume> y =
        std::make_shared<GraphicsVolume>(graphics_lib_);
    r = y->Create(width, height, depth, num_of_components, byte_width, border);
    assert(r);
    if (!r)
        return false;

    std::shared_ptr<GraphicsVolume> z =
        std::make_shared<GraphicsVolume>(graphics_lib_);
    r = z->Create(width, height, depth, num_of_components, byte_width, border);
    assert(r);
    if (!r)
        return false;

    x_ = x;
    y_ = y;
    z_ = z;
    return true;
}

void GraphicsVolume3::Swap(GraphicsVolume3& obj)
{
    std::shared_ptr<GraphicsVolume> x = obj.x();
    std::shared_ptr<GraphicsVolume> y = obj.y();
    std::shared_ptr<GraphicsVolume> z = obj.z();

    obj.Assign(x_, y_, z_);
    Assign(x, y, z);
}

void GraphicsVolume3::Swap(std::shared_ptr<GraphicsVolume>* x,
                           std::shared_ptr<GraphicsVolume>* y,
                           std::shared_ptr<GraphicsVolume>* z)
{
    std::shared_ptr<GraphicsVolume> ¦Á = *x;
    std::shared_ptr<GraphicsVolume> ¦Â = *y;
    std::shared_ptr<GraphicsVolume> ¦Ã = *z;

    *x = x_;
    *y = y_;
    *z = z_;
    Assign(¦Á, ¦Â, ¦Ã);
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
