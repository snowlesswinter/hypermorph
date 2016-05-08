#include "stdafx.h"
#include "trackball.h"

#include <vmath.hpp>

using namespace vmath;

class TrackballImpl : public Trackball
{
public:
    TrackballImpl(int width, int height, float radius);

    virtual void MouseDown(int x, int y) override;
    virtual void MouseUp(int x, int y) override;
    virtual void MouseMove(int x, int y) override;
    virtual void MouseWheel(int x, int y, float delta) override;
    virtual void ReturnHome() override;
    virtual vmath::Matrix3 GetRotation() const override;
    virtual float GetZoom() const override;
    virtual void Update(uint32_t microseconds) override;
    virtual void OnViewportSized(int width, int height) override;

private:
    vmath::Vector3 MapToSphere(int x, int y);

    vmath::Vector3 anchor_pos_;
    vmath::Vector3 current_pos_;
    vmath::Vector3 previous_pos_;
    vmath::Vector3 axis_;
    vmath::Quat anchor_quat_;
    bool active_;
    float radius_;
    float radians_per_second_;
    float distance_per_second_;
    int width_;
    int height_;
    float zoom_;
    float start_zoom_;
    int start_y_;
    uint32_t current_time_;
    uint32_t previous_time_;

    struct VoyageHome
    {
        bool active_;
        vmath::Quat departure_quat_;
        float departure_zoom_;
        uint32_t microseconds_;

        VoyageHome()
            : active_(false)
            , departure_quat_()
            , departure_zoom_(0.0f)
            , microseconds_(0)
        {
        }
    } voyage_home_;

    struct Inertia
    {
        bool active_;
        vmath::Vector3 axis_;
        float radians_per_second_;
        float distance_per_second_;

        Inertia()
            : active_(false)
            , axis_()
            , radians_per_second_(0.0f)
            , distance_per_second_(0.0f)
        {
        }
    } inertia_;
};

TrackballImpl::TrackballImpl(int width, int height, float radius)
    : anchor_pos_(Vector3(0))
    , current_pos_(Vector3(0))
    , previous_pos_(Vector3(0))
    , axis_(Vector3(1))
    , anchor_quat_(vmath::Quat::identity())
    , active_(false)
    , radius_(radius)
    , radians_per_second_(0.0f)
    , distance_per_second_(0.0f)
    , width_(width)
    , height_(height)
    , zoom_(0)
    , start_zoom_(0)
    , start_y_(0)
    , current_time_(0)
    , previous_time_(0)
    , voyage_home_()
    , inertia_()
{
}

void TrackballImpl::MouseDown(int x, int y)
{
    radians_per_second_ = 0;
    distance_per_second_ = 0;
    previous_pos_ = current_pos_ = anchor_pos_ = MapToSphere(x, y);
    active_ = true;
    start_zoom_ = zoom_;
    start_y_ = y;
}

void TrackballImpl::MouseUp(int x, int y)
{
    if (active_) {
        float deltaDistance = (y - start_y_) * 0.01f;
        zoom_ = start_zoom_ + deltaDistance;
        start_zoom_ = zoom_;
        start_y_ = y;
        active_ = false;
    }

    // Calculate via definition:
    // 
    // Vector3 axis = cross(anchor_pos_, current_pos_);
    // Vector3 n_axis = normalize(axis);
    // float radians = atan2f(length(axis), dot(anchor_pos_, current_pos_));
    // Quat q(n_axis * sinf(radians / 2), cosf(radians / 2));

    Quat q = Quat::rotation(anchor_pos_, current_pos_);
    anchor_quat_ = rotate(q, anchor_quat_);

    if (radians_per_second_ > 0 || distance_per_second_ != 0) {
        inertia_.active_ = true;
        inertia_.radians_per_second_ = radians_per_second_;
        inertia_.distance_per_second_ = distance_per_second_;
        inertia_.axis_ = axis_;
    }
}

void TrackballImpl::MouseMove(int x, int y)
{
    current_pos_ = MapToSphere(x, y);

    float radians = acos(dot(previous_pos_, current_pos_));
    uint32_t microseconds = current_time_ - previous_time_;
    
    if (radians > 0.01f && microseconds > 0) {
        radians_per_second_ = 1000000.0f * radians / microseconds;
        axis_ = normalize(cross(previous_pos_, current_pos_));
    } else {
        radians_per_second_ = 0;
    }

    start_zoom_ = zoom_;
    start_y_ = y;

    previous_pos_ = current_pos_;
    previous_time_ = current_time_;
}

Matrix3 TrackballImpl::GetRotation() const
{
    if (!active_) {
        return Matrix3(anchor_quat_);
    }

    Quat q = Quat::rotation(anchor_pos_, current_pos_);
    return Matrix3(rotate(q, anchor_quat_));
}

Vector3 TrackballImpl::MapToSphere(int x, int y)
{
    y = static_cast<int>(height_) - y; // Note that the y-coordinate of the
                                       // window is towards down.
    const float safe_radius = radius_ * 0.9999999f;
    float tx = x - static_cast<float>(width_) / 2.0f;
    float ty = y - static_cast<float>(height_) / 2.0f;

    float d_square = tx * tx + ty * ty;

    bool use_holroyd_method = true;
    if (use_holroyd_method) {
        if (d_square <= safe_radius * safe_radius / 2.0f) {
            float z = sqrt(radius_ * radius_ - d_square);
            return normalize(Vector3(tx, ty, z));
        }

        float z = radius_ * radius_ / (2.0f * sqrtf(d_square));
        return normalize(Vector3(tx, ty, z));
    } else {
        // Shoemake's method.
        if (d_square > safe_radius * safe_radius) {
            float theta = atan2(ty, tx);
            tx = safe_radius * cos(theta);
            ty = safe_radius * sin(theta);

            d_square = tx * tx + ty * ty;
        }

        float z = sqrt(radius_ * radius_ - d_square);
        return Vector3(tx, ty, z) / radius_;
    }
}

void TrackballImpl::Update(uint32_t microseconds)
{
    current_time_ += microseconds;

    if (voyage_home_.active_) {
        voyage_home_.microseconds_ += microseconds;
        float t = voyage_home_.microseconds_ / 200000.0f;
        
        if (t > 1) {
            anchor_quat_ = Quat::identity();
            start_zoom_ = zoom_ = 0;
            voyage_home_.active_ = false;
            return;
        }

        anchor_quat_ = slerp(t, voyage_home_.departure_quat_, Quat::identity());
        start_zoom_ = zoom_ = voyage_home_.departure_zoom_ * (1-t);
        inertia_.active_ = false;
    }

    if (inertia_.active_) {
        inertia_.radians_per_second_ -= 0.00001f * microseconds;

        if (inertia_.radians_per_second_ < 0) {
            radians_per_second_ = 0;
        } else {
            Quat q = Quat::rotation(
                inertia_.radians_per_second_ * microseconds * 0.000001f,
                inertia_.axis_);
            anchor_quat_ = rotate(q, anchor_quat_);
        }

        inertia_.distance_per_second_ *= 0.75f;

        if (fabs(inertia_.distance_per_second_) < 0.0001f) {
            distance_per_second_ = 0.0f;
        } else {
            zoom_ += distance_per_second_ * 0.001f;
        }

        if (fabs(inertia_.distance_per_second_) < 0.0001f &&
                inertia_.radians_per_second_ < 0.0f)
            inertia_.active_ = false;
    }
}

void TrackballImpl::OnViewportSized(int width, int height)
{
    width_ = width;
    height_ = height;
}

void TrackballImpl::ReturnHome()
{
    voyage_home_.active_ = true;
    voyage_home_.departure_quat_ = anchor_quat_;
    voyage_home_.departure_zoom_ = zoom_;
    voyage_home_.microseconds_ = 0;
}

float TrackballImpl::GetZoom() const
{
    return zoom_;
}

Trackball* Trackball::CreateTrackball(int width, int height, float radius)
{
    return new TrackballImpl(width, height, radius);
}

void TrackballImpl::MouseWheel(int x, int y, float delta)
{
    zoom_ += delta;
}