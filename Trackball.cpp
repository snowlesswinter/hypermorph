#include "Utility.h"

using namespace vmath;

class Trackball : public ITrackball {
    public:
        Trackball(float width, float height, float radius);
        void MouseDown(int x, int y);
        void MouseUp(int x, int y);
        void MouseMove(int x, int y);
        void ReturnHome();
        vmath::Matrix3 GetRotation() const;
        void Update(unsigned int microseconds);
        float GetZoom() const;
    private:
        vmath::Vector3 MapToSphere(int x, int y);
        vmath::Vector3 m_startPos;
        vmath::Vector3 m_currentPos;
        vmath::Vector3 m_previousPos;
        vmath::Vector3 m_axis;
        vmath::Quat m_quat;
        bool m_active;
        float m_radius;
        float m_radiansPerSecond;
        float m_distancePerSecond;
        float m_width;
        float m_height;
        float m_zoom;
        float m_startZoom;
        int m_startY;
        unsigned int m_currentTime;
        unsigned int m_previousTime;

        struct VoyageHome {
            bool Active;
            vmath::Quat DepartureQuat;
            float DepartureZoom;
            unsigned int microseconds;
        } m_voyageHome;

        struct Inertia {
            bool Active;
            vmath::Vector3 Axis;
            float RadiansPerSecond;
            float DistancePerSecond;
        } m_inertia;
};

Trackball::Trackball(float width, float height, float radius)
{
    m_currentTime = 0;
    m_inertia.Active = false;
    m_voyageHome.Active = false;
    m_active = false;
    m_quat = vmath::Quat::identity();
    m_radius = radius;
    m_startPos = m_currentPos = m_previousPos = Vector3(0);
    m_width = width;
    m_height = height;
    m_startZoom = m_zoom = 0;
}

void Trackball::MouseDown(int x, int y)
{
    m_radiansPerSecond = 0;
    m_distancePerSecond = 0;
    m_previousPos = m_currentPos = m_startPos = MapToSphere(x, y);
    m_active = true;
    m_startZoom = m_zoom;
    m_startY = y;
}

void Trackball::MouseUp(int x, int y)
{
    if (m_active) {
        float deltaDistance = (y - m_startY) * 0.01f;
        m_zoom = m_startZoom + deltaDistance;
        m_startZoom = m_zoom;
        m_startY = y;
        m_active = false;
    }

    Quat q = Quat::rotation(m_startPos, m_currentPos);
    m_quat = rotate(q, m_quat);

    if (m_radiansPerSecond > 0 || m_distancePerSecond != 0) {
        m_inertia.Active = true;
        m_inertia.RadiansPerSecond = m_radiansPerSecond;
        m_inertia.DistancePerSecond = m_distancePerSecond;
        m_inertia.Axis = m_axis;
    }
}

void Trackball::MouseMove(int x, int y)
{
    m_currentPos = MapToSphere(x, y);

    float radians = acos(dot(m_previousPos, m_currentPos));
    unsigned int microseconds = m_currentTime - m_previousTime;
    
    if (radians > 0.01f && microseconds > 0) {
        m_radiansPerSecond = 1000000.0f * radians / microseconds;
        m_axis = normalize(cross(m_previousPos, m_currentPos));
    } else {
        m_radiansPerSecond = 0;
    }
/*
    if (m_active) {
        float deltaDistance = (y - m_startY) * 0.01f;
        if (std::abs(deltaDistance) > 0.03f && microseconds > 0) {
            m_distancePerSecond = 1000000.0f * deltaDistance / microseconds;
        } else {
            m_distancePerSecond = 0;
        }

        m_zoom = m_startZoom + deltaDistance;
    }
*/
    m_startZoom = m_zoom;
    m_startY = y;

    m_previousPos = m_currentPos;
    m_previousTime = m_currentTime;
}

Matrix3 Trackball::GetRotation() const
{
    if (!m_active)
        return Matrix3(m_quat);

    Quat q = Quat::rotation(m_startPos, m_currentPos);
    return Matrix3(rotate(q, m_quat));
}

Vector3 Trackball::MapToSphere(int x, int y)
{
    x = int(m_width) - x;
    const float SafeRadius = m_radius * 0.99f;
    float fx = x - m_width / 2.0f;
    float fy = y - m_height / 2.0f;

    float lenSqr = fx*fx+fy*fy;
    
    if (lenSqr > SafeRadius*SafeRadius) {
        float theta = atan2(fy, fx);
        fx = SafeRadius * cos(theta);
        fy = SafeRadius * sin(theta);
    }
    
    lenSqr = fx*fx+fy*fy;
    float z = sqrt(m_radius*m_radius - lenSqr);
    return Vector3(fx, fy, z) / m_radius;
}

void Trackball::Update(unsigned int microseconds)
{
    m_currentTime += microseconds;

    if (m_voyageHome.Active) {
        m_voyageHome.microseconds += microseconds;
        float t = m_voyageHome.microseconds / 200000.0f;
        
        if (t > 1) {
            m_quat = Quat::identity();
            m_startZoom = m_zoom = 0;
            m_voyageHome.Active = false;
            return;
        }

        m_quat = slerp(t, m_voyageHome.DepartureQuat, Quat::identity());
        m_startZoom = m_zoom = m_voyageHome.DepartureZoom * (1-t);
        m_inertia.Active = false;
    }

    if (m_inertia.Active) {
        m_inertia.RadiansPerSecond -= 0.00001f * microseconds;

        if (m_inertia.RadiansPerSecond < 0) {
            m_radiansPerSecond = 0;
        } else {
            Quat q = Quat::rotation(m_inertia.RadiansPerSecond * microseconds * 0.000001f, m_inertia.Axis);
            m_quat = rotate(q, m_quat);
        }

        m_inertia.DistancePerSecond *= 0.75f;

        if (fabs(m_inertia.DistancePerSecond) < 0.0001f) {
            m_distancePerSecond = 0.0f;
        } else {
            m_zoom += m_distancePerSecond * 0.001f;
        }

        if (fabs(m_inertia.DistancePerSecond) < 0.0001f && m_inertia.RadiansPerSecond < 0.0f)
            m_inertia.Active = false;
    }
}

void Trackball::ReturnHome()
{
    m_voyageHome.Active = true;
    m_voyageHome.DepartureQuat = m_quat;
    m_voyageHome.DepartureZoom = m_zoom;
    m_voyageHome.microseconds = 0;
}

float Trackball::GetZoom() const
{
    return m_zoom;
}

ITrackball* CreateTrackball(float width, float height, float radius)
{
    return new Trackball(width, height, radius);
}
