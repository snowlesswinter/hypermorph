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

#include <algorithm>
#include <array>
#include <cassert>
#include <ctime>

#include "third_party/glm/geometric.hpp"
#include "third_party/glm/vec3.hpp"

namespace
{
const float g_pi = std::acos(-1.0f);

float QuadraticEasingIn(float start_pos, float distance, float time_elapsed,
                        float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * ratio * ratio;
}

float QuadraticEasingOut(float start_pos, float distance, float time_elapsed,
                         float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * ratio * (2.0f - ratio);
}

float QuadraticEasingInOut(float start_pos, float distance, float time_elapsed,
                           float duration)
{
    float ratio = 2.0f * time_elapsed / duration;
    if (ratio < 1.0f)
        return start_pos + distance * ratio * ratio * 0.5f;

    ratio -= 1.0f;
    return start_pos + distance * (1.0f - ratio * (ratio - 2.0f)) * 0.5f;
}

float CubicEasingIn(float start_pos, float distance, float time_elapsed,
                           float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * ratio * ratio * ratio;
}

float CubicEasingOut(float start_pos, float distance, float time_elapsed,
                           float duration)
{
    float ratio = time_elapsed / duration;
    ratio -= 1.0f;
    return start_pos + distance * (ratio * ratio * ratio + 1.0f);
}

float CubicEasingInOut(float start_pos, float distance, float time_elapsed,
                           float duration)
{
    float ratio = 2.0f * time_elapsed / duration;
    if (ratio < 1.0f)
        return start_pos + distance * ratio * ratio * ratio * 0.5f;

    ratio -= 2.0f;
    return start_pos + distance * (ratio * ratio * ratio + 2.0f) * 0.5f;
}

float QuarticEasingIn(float start_pos, float distance, float time_elapsed,
                       float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * ratio * ratio * ratio * ratio;
}

float QuarticEasingOut(float start_pos, float distance, float time_elapsed,
                         float duration)
{
    float ratio = time_elapsed / duration;
    ratio -= 1.0f;
    return start_pos + distance * (1.0f - ratio * ratio * ratio * ratio);
}

float QuarticEasingInOut(float start_pos, float distance, float time_elapsed,
                         float duration)
{
    float ratio = 2.0f * time_elapsed / duration;
    if (ratio < 1.0f)
        return start_pos + distance * ratio * ratio * ratio * ratio * 0.5f;

    ratio -= 2.0f;
    return start_pos + distance * (2.0f - ratio * ratio * ratio * ratio) * 0.5f;
}

float SinusoidalEasingIn(float start_pos, float distance, float time_elapsed,
                         float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * (1.0f - std::cos(ratio * 0.5f * g_pi));
}

float SinusoidalEasingOut(float start_pos, float distance, float time_elapsed,
                            float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * std::sin(ratio * 0.5f * g_pi);
}

float SinusoidalEasingInOut(float start_pos, float distance, float time_elapsed,
                            float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * (1.0f - std::cos(ratio * g_pi)) * 0.5f;
}

float CircularEasingIn(float start_pos, float distance, float time_elapsed,
                         float duration)
{
    float ratio = time_elapsed / duration;
    return start_pos + distance * (1.0f - std::sqrt(1.0f - ratio * ratio));
}

float CircularEasingOut(float start_pos, float distance, float time_elapsed,
                            float duration)
{
    float ratio = time_elapsed / duration;
    ratio -= 1.0f;
    return start_pos + distance * std::sqrt(1.0f - ratio * ratio);
}

float CircularEasingInOut(float start_pos, float distance, float time_elapsed,
                            float duration)
{
    float ratio = 2.0f * time_elapsed / duration;
    if (ratio < 1.0f)
        return start_pos +
            distance * (1.0f - std::sqrt(1.0f - ratio * ratio)) * 0.5f;

    ratio -= 2.0f;
    return start_pos +
        distance * (1.0f + std::sqrt(1.0f - ratio * ratio)) * 0.5f;
}

typedef float(__cdecl *AnimateFunc)(float, float, float, float);
AnimateFunc g_ani_func[] = {
    CubicEasingIn,
    CubicEasingOut,
    CubicEasingInOut,
    QuarticEasingIn,
    QuarticEasingOut,
    QuarticEasingInOut,
    SinusoidalEasingIn,
    SinusoidalEasingOut,
    SinusoidalEasingInOut,
    CircularEasingIn,
    CircularEasingOut,
    CircularEasingInOut,
};

static const int g_num_func = sizeof(g_ani_func) / sizeof(*g_ani_func);
}



class Scene::Dancer
{
public:
    Dancer();
    ~Dancer();

    void Init();
    void Step(float time_step);

    const glm::vec3& position() const { return position_; }

private:
    float ChooseTarget(float pos);

    glm::vec3 position_;
    glm::vec3 target_;
    glm::vec3 velocity_;
    glm::vec3 start_pos_;
    glm::vec3 distance_;
    float time_elapsed_;
    float animation_duration_;
    float slow_motion_count_down_;
    std::array<AnimateFunc, 3> func_;
};

Scene::Dancer::Dancer()
    : position_(0.0f)
    , target_()
    , velocity_(0.08f)
    , start_pos_(0.0f)
    , distance_(0.0f)
    , time_elapsed_(0.0f)
    , animation_duration_(2.0f)
    , slow_motion_count_down_(0.0f)
    , func_({QuadraticEasingIn, QuadraticEasingIn, QuadraticEasingIn})
{
}

Scene::Dancer::~Dancer()
{
}

void Scene::Dancer::Init()
{
    std::srand(0x77703331);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    target_ = glm::vec3(ChooseTarget(position_.x), ChooseTarget(position_.y),
                        ChooseTarget(position_.z));
    distance_ = target_ - position_;
    animation_duration_ = glm::length(distance_) * 4.0f;

    for (auto i = func_.begin(); i != func_.end(); i++)
        *i = g_ani_func[(rand() % g_num_func)];
}

void Scene::Dancer::Step(float time_step)
{
    time_elapsed_ += time_step;
    if (time_elapsed_ >= animation_duration_) {
        target_ = glm::vec3(ChooseTarget(position_.x),
                            ChooseTarget(position_.y),
                            ChooseTarget(position_.z));

        time_elapsed_ = 0.0f;
        distance_ = target_ - position_;
        animation_duration_ = glm::length(distance_) * 4.0f;
        start_pos_ = position_;

        for (auto i = func_.begin(); i != func_.end(); i++)
            *i = g_ani_func[(rand() % g_num_func)];
    }

    float ratio = time_elapsed_ / animation_duration_;
    glm::vec3 dir = glm::normalize(target_ - position_);
    for (int i = 0; i < 3; i++)
        position_[i] = func_[i](start_pos_[i], distance_[i], time_elapsed_,
                                animation_duration_);
}

float Scene::Dancer::ChooseTarget(float pos)
{
    int r = std::rand() % 1000;
    float d = 0.5f;
    int k = static_cast<int>(d * 1000);
    float p;
    if (r > k) {
        float t = static_cast<float>(r - k) / (1000.0f - k);
        p = (pos > 0.5f ? 0.0f : 0.5f) + t * 0.5f;
    } else {
        float t = static_cast<float>(r) / k;
        float p = (pos > 0.5f ? 0.5f : 0.0f) + t * 0.5f;
    }
    p = std::max(std::min(p, 0.9f), 0.1f);
    return p;
}

Scene::Scene()
    : dance_(std::make_shared<Dancer>())
{
}

Scene::~Scene()
{
}

void Scene::Advance(float time_step)
{
    dance_->Step(time_step);
}

bool Scene::Init()
{
    dance_->Init();
    return true;
}

glm::vec3 Scene::GetDancerPos() const
{
    return dance_->position();
}
