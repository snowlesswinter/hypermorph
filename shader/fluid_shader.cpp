#include "stdafx.h"
#include "fluid_shader.h"

std::string FluidShader::GetVertexShaderCode()
{
    return R"(
in vec4 Position;
out int vInstance;

void main()
{
    gl_Position = Position;
    vInstance = gl_InstanceID;
}
)";
}

std::string FluidShader::GetPickLayerShaderCode()
{
    return R"(
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
 
in int vInstance[3];
out float gLayer;
 
uniform float InverseSize;
 
void main()
{
    gl_Layer = vInstance[0];
    gLayer = float(gl_Layer) + 0.5;
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();
    EndPrimitive();
}
)";
}

std::string FluidShader::GetFillShaderCode()
{
    return R"(
out vec3 FragColor;

void main()
{
    FragColor = vec3(1, 0, 0);
}
)";
}

std::string FluidShader::GetAvectShaderCode()
{
    // NOTE: Think of the real world, smoke and air molecules would hit each
    //       other. The further the smoke traveled, the more hitting occurred.
    //       So I think the real dissipation of velocity should be proportional
    //       to the distance the voxel moved within the time slice(maybe
    //       non-linear, I don't know). And that could be a key why the fluid's
    //       behavior changed so much if we shorten the time step by a level.
    //
    //       Since the amount of calculation introduced by a more accurate
    //       algorithm would be significantly larger(computing square root of 3
    //       velocities in different direction, I decided to use a constant
    //       dissipation for performance consideration.

    return R"(
out vec3 FragColor;

uniform sampler3D VelocityTexture;
uniform sampler3D SourceTexture;
uniform sampler3D Obstacles;

uniform vec3 InverseSize;
uniform float TimeStep;
uniform float Dissipation;

in float gLayer;

void main()
{
    vec3 fragCoord = vec3(gl_FragCoord.xy, gLayer);
    vec3 u = texture(VelocityTexture, InverseSize * fragCoord).xyz;

    vec3 coord = fragCoord - TimeStep * u;

    FragColor = Dissipation * texture(SourceTexture, InverseSize * coord).xyz;
    return;

    // Boundary check
    ivec3 tex_size = textureSize(VelocityTexture, 0);
    ivec3 icoord = ivec3(coord);
    if ((icoord.y > tex_size.y - 1) || (icoord.y < 0) ||
            (icoord.x > tex_size.x - 1) || (icoord.x < 0) ||
            (icoord.z > tex_size.z - 1) || (icoord.z < 0)) {
        // Why not FragColor = vec4(0); ????
        FragColor = Dissipation * texture(SourceTexture, InverseSize * coord).xyz;
    } else {
        FragColor = Dissipation * texture(SourceTexture, InverseSize * coord).xyz;
    }
}
)";
}

std::string FluidShader::GetJacobiShaderCode()
{
    return R"(
out vec3 FragColor;

uniform sampler3D Pressure;
uniform sampler3D Divergence;
uniform sampler3D Obstacles;

uniform float Alpha;
uniform float InverseBeta;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring pressure:
    float pN = texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0)).r;
    float pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0)).r;
    float pE = texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0)).r;
    float pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0)).r;
    float pU = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1)).r;
    float pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1)).r;
    float pC = texelFetch(Pressure, T, 0).r;

    // Handle boundary problem
    // Use center pressure for solid cells
    ivec3 tex_size = textureSize(Pressure, 0);
    if (T.y >= tex_size.y - 1)
        pN = pC;

    if (T.y <= 0)
        pS = pC;

    if (T.x >= tex_size.x - 1)
        pE = pC;

    if (T.x <= 0)
        pW = pC;

    if (T.z >= tex_size.z - 1)
        pU = pC;

    if (T.z <= 0)
        pD = pC;

    float bC = texelFetch(Divergence, T, 0).r;
    FragColor = vec3((pW + pE + pS + pN + pU + pD + Alpha * bC) * InverseBeta,
                     0.0f, 0.0f);
}
)";
}

std::string FluidShader::GetDampedJacobiShaderCode()
{
    return R"(
out vec3 FragColor;

uniform sampler3D Pressure;
uniform sampler3D Divergence;
uniform sampler3D Obstacles;

uniform float one_minus_omega;
uniform float Alpha;
uniform float InverseBeta;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring pressure:
    float pN = texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0)).r;
    float pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0)).r;
    float pE = texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0)).r;
    float pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0)).r;
    float pU = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1)).r;
    float pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1)).r;
    float pC = texelFetch(Pressure, T, 0).r;

    // Handle boundary problem
    // Use center pressure for solid cells
    ivec3 tex_size = textureSize(Pressure, 0);
    if (T.y >= tex_size.y - 1)
        pN = pC;

    if (T.y <= 0)
        pS = pC;

    if (T.x >= tex_size.x - 1)
        pE = pC;

    if (T.x <= 0)
        pW = pC;

    if (T.z >= tex_size.z - 1)
        pU = pC;

    if (T.z <= 0)
        pD = pC;

    float bC = texelFetch(Divergence, T, 0).r;
    FragColor = vec3(
        one_minus_omega * pC +
            (pW + pE + pS + pN + pU + pD + Alpha * bC) * InverseBeta,
        0.0f, 0.0f);
}
)";
}

std::string FluidShader::GetComputeDivergenceShaderCode()
{
    return R"(
out float FragColor;

uniform sampler3D Velocity;
uniform sampler3D Obstacles;
uniform float HalfInverseCellSize;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring velocities:
    vec3 vN = texelFetchOffset(Velocity, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 vS = texelFetchOffset(Velocity, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 vE = texelFetchOffset(Velocity, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 vW = texelFetchOffset(Velocity, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 vU = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 vD = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, -1)).xyz;
    vec3 vC = texelFetch(Velocity, T, 0).xyz;

    float diff_ew = vE.x - vW.x;
    float diff_ns = vN.y - vS.y;
    float diff_ud = vU.z - vD.z;

    // Handle boundary problem
    ivec3 tex_size = textureSize(Velocity, 0);
    if (T.x >= tex_size.x - 1)
        diff_ew = -vC.x - vW.x;

    if (T.x <= 0)
        diff_ew = vE.x + vC.x;

    if (T.y >= tex_size.y - 1)
        diff_ns = -vC.y - vS.y;

    if (T.y <= 0)
        diff_ns = vN.y + vC.y;

    if (T.z >= tex_size.z - 1)
        diff_ud = -vC.z - vD.z;

    if (T.z <= 0)
        diff_ud = vU.z + vC.z;

    FragColor = HalfInverseCellSize * (diff_ew + diff_ns + diff_ud);
}
)";
}

std::string FluidShader::GetSubtractGradientShaderCode()
{
    return R"(
out vec3 FragColor;

uniform sampler3D Velocity;
uniform sampler3D Pressure;
uniform sampler3D Obstacles;
uniform float GradientScale;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring pressure:
    float pN = texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0)).r;
    float pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0)).r;
    float pE = texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0)).r;
    float pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0)).r;
    float pU = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1)).r;
    float pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1)).r;
    float pC = texelFetch(Pressure, T, 0).r;

    // Handle boundary problem
    // Use center pressure for solid cells:
    vec3 vMask = vec3(1);

    ivec3 tex_size = textureSize(Pressure, 0);
    if (T.y >= tex_size.y - 1) {
        pN = pC;
        vMask.y = 0;
    }

    if (T.y <= 0) {
        pS = pC;
        vMask.y = 0;
    }

    if (T.x >= tex_size.x - 1) {
        pE = pC;
        vMask.x = 0;
    }

    if (T.x <= 0) {
        pW = pC;
        vMask.x = 0;
    }

    if (T.z >= tex_size.z - 1) {
        pU = pC;
        vMask.z = 0;
    }

    if (T.z <= 0) {
        pD = pC;
        vMask.z = 0;
    }

    // Enforce the free-slip boundary condition:
    vec3 oldV = texelFetch(Velocity, T, 0).xyz;
    vec3 grad = vec3(pE - pW, pN - pS, pU - pD) * GradientScale;
    vec3 newV = oldV - grad;
    FragColor = vMask * newV;
}
)";
}

std::string FluidShader::GetSplatShaderCode()
{
    return R"(
out vec4 FragColor;

uniform vec3 center_point;
uniform vec3 hotspot;
uniform float Radius;
uniform vec3 FillColor;

in float gLayer;

void main()
{
    if (gl_FragCoord.y > 1 && gl_FragCoord.y < 3) {
        float d = distance(center_point.xz, vec2(gl_FragCoord.x, gLayer));
        if (d < Radius) {
            float scale = (Radius - distance(hotspot.xz, vec2(gl_FragCoord.x, gLayer))) / Radius;
            scale = max(scale, 0.5);
            FragColor = vec4(scale * FillColor, 1.0);
            return;
        }
    }

    FragColor = vec4(0);
}
)";
}

std::string FluidShader::GetBuoyancyShaderCode()
{
    // In the original implementation, density is also accounted for the
    // acceleration. But I don't think that's reasonable, so I removed this
    // factor for gravity calculation.
    return R"(
out vec3 FragColor;
uniform sampler3D Velocity;
uniform sampler3D Temperature;
uniform float AmbientTemperature;
uniform float TimeStep;
uniform float Sigma;
uniform float Kappa;

in float gLayer;

void main()
{
    ivec3 TC = ivec3(gl_FragCoord.xy, gLayer);
    float t = texelFetch(Temperature, TC, 0).r;
    vec3 v = texelFetch(Velocity, TC, 0).xyz;

    FragColor = v;

    if (t > AmbientTemperature) {
        FragColor += TimeStep * ((t - AmbientTemperature) * Sigma - Kappa ) * vec3(0, 1, 0);
    }
}
)";
}