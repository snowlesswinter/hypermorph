#include "stdafx.h"
#include "raycast_shader.h"

std::string RaycastShader::Vertex()
{
    return R"(
in vec4 Position;
out vec4 vPosition;
uniform mat4 ModelviewProjection;

void main()
{
    gl_Position = ModelviewProjection * Position;
    vPosition = Position;
}
)";
}

std::string RaycastShader::Geometry()
{
    return R"(
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

in vec4 vPosition[1];

uniform mat4 ModelviewProjection;
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;
uniform mat4 Modelview;

vec4 objCube[8]; // Object space coordinate of cube corner
vec4 ndcCube[8]; // Normalized device coordinate of cube corner
ivec4 faces[6];  // Vertex indices of the cube faces

void emit_vert(int vert)
{
    gl_Position = ndcCube[vert];
    EmitVertex();
}

void emit_face(int i)
{
    emit_vert(faces[i][0]);
    emit_vert(faces[i][1]);
    emit_vert(faces[i][2]);
    emit_vert(faces[i][3]);
    EndPrimitive();
}

void main()
{
    faces[0] = ivec4(0,1,3,2);
    faces[1] = ivec4(5,4,6,7);
    faces[2] = ivec4(4,5,0,1);
    faces[3] = ivec4(3,2,7,6);
    faces[4] = ivec4(0,3,4,7);
    faces[5] = ivec4(2,1,6,5);

    vec4 P = vPosition[0];
    vec4 I = vec4(1,0,0,0);
    vec4 J = vec4(0,1,0,0);
    vec4 K = vec4(0,0,1,0);

    objCube[0] = P+K+I+J;
    objCube[1] = P+K+I-J;
    objCube[2] = P+K-I-J;
    objCube[3] = P+K-I+J;
    objCube[4] = P-K+I+J;
    objCube[5] = P-K+I-J;
    objCube[6] = P-K-I-J;
    objCube[7] = P-K-I+J;

    // Transform the corners of the box:
    for (int vert = 0; vert < 8; vert++)
        ndcCube[vert] = ModelviewProjection * objCube[vert];

    // Emit the six faces:
    for (int face = 0; face < 6; face++)
        emit_face(face);
}
)";
}

std::string RaycastShader::Fragment()
{
    return R"(
out vec4 FragColor;

uniform sampler3D Density;
uniform vec3 LightPosition = vec3(1.0, 1.0, 2.0);
//uniform vec3 LightIntensity = vec3(6.2109375, 7.2265625, 8.0078125);
uniform vec3 LightIntensity = vec3(10.0);
uniform float Absorption = 10.0;
uniform mat4 Modelview;
uniform float FocalLength;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;

const float maxDist = sqrt(2.0);
const int numSamples = 128;
const float stepSize = maxDist / float(numSamples);
const int numLightSamples = 32;
const float lscale = maxDist / float(numLightSamples);
const float densityFactor = 10;

float GetDensity(vec3 pos)
{
    return texture(Density, pos).x * densityFactor;
}

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, AABB aabb, out float near, out float far)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    near = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    far = min(t.x, t.y);
    return near <= far;
}

void main()
{
    vec3 rayDirection;

    // Normalized ray direction vector in world space.
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize - 1.0;
    rayDirection.z = -FocalLength;

    // Transform the ray direction vector to model space?
    rayDirection = (vec4(rayDirection, 0) * Modelview).xyz;

    // Ray origin is already in model space.
    Ray eye = Ray( RayOrigin, normalize(rayDirection) );
    AABB aabb = AABB(vec3(-1.0), vec3(+1.0));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0)
        tnear = 0.0;

    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 step = normalize(rayStop-rayStart) * stepSize;
    float travel = distance(rayStop, rayStart);
    float T = 1.0;
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < numSamples && travel > 0.0; ++i, pos += step, travel -= stepSize) {

        float density = GetDensity(pos);
        if (density <= 0.0)
            continue;

        T *= 1.0 - density * stepSize * Absorption;
        if (T <= 0.01) {
            break;
        }

        vec3 lightDir = normalize(LightPosition - pos) * lscale;
        float Tl = 1.0;
        vec3 lpos = pos + lightDir;

        for (int s = 0; s < numLightSamples; ++s) {
            float ld = texture(Density, lpos).x;
            Tl *= 1.0 - Absorption * stepSize * ld;
            if (Tl <= 0.01) 
                lpos += lightDir;
        }

        vec3 Li = LightIntensity * Tl;
        Lo += Li * T * density * stepSize;
    }

    FragColor.rgb = Lo;
    FragColor.a = 1 - T;
}
)";
}