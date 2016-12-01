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
#include "scene.h"

#include <cassert>

#include "third_party/glm/geometric.hpp"
#include "third_party/glm/vec3.hpp"

class Scene::Dancer
{
public:
    Dancer();
    ~Dancer();

    void Init(const glm::vec3& grid_size);
    void Step(float time_step);

private:
    float ChooseTarget(float pos);

    glm::vec3 grid_size_;
    glm::vec3 position_;
    glm::vec3 target_;
    glm::vec3 velocity_;
};

Scene::Dancer::Dancer()
    : grid_size_()
    , position_()
    , target_()
    , velocity_(0.05f)
{
}

Scene::Dancer::~Dancer()
{

}

void Scene::Dancer::Init(const glm::vec3& grid_size)
{
    grid_size_ = grid_size;
    std::srand(0x77773331);
}

void Scene::Dancer::Step(float time_step)
{
    if (glm::distance(target_, position_) < 0.001f) {
        target_ = glm::vec3(ChooseTarget(position_.x),
                            ChooseTarget(position_.y),
                            ChooseTarget(position_.z));
    }

    glm::vec3 dir = glm::normalize(target_ - position_);
    position_ += dir * velocity_ * time_step;
}

float Scene::Dancer::ChooseTarget(float pos)
{
    int r = std::rand() % 1000;
    if (r > 333) {
        float t = static_cast<float>(r - 333) / 666.0f;
        return (pos > 0.5f ? 0.0f : 0.5f) + t * 0.5f;
    } else {
        float t = static_cast<float>(r) / 333.0f;
        return (pos > 0.5f ? 0.5f : 0.0f) + t * 0.5f;
    }
}

Scene::Scene()
    : dance_(std::make_shared<Dancer>())
{
}

Scene::~Scene()
{
}
