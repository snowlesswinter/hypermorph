#include "stdafx.h"
#include "multigrid_shader.h"

#include <regex>

namespace
{
const char* kRestrictCore = R"(
    vec3 f_coord = vec3(gl_FragCoord.xy, gLayer) * 2.0f;
    vec3 c = inverse_size * f_coord;

    ivec3 t00 = ivec3(1, 1, -1); 
    ivec3 t01 = ivec3(0, 1, -1); 
    ivec3 t02 = ivec3(-1, 1, -1);
    ivec3 t03 = ivec3(1, 0, -1); 
    ivec3 t04 = ivec3(0, 0, -1); 
    ivec3 t05 = ivec3(-1, 0, -1);
    ivec3 t06 = ivec3(1, -1, -1);
    ivec3 t07 = ivec3(0, -1, -1);
    ivec3 t08 = ivec3(-1, -1, -1);

    ivec3 t09 = ivec3(1, 1, 0); 
    ivec3 t10 = ivec3(0, 1, 0); 
    ivec3 t11 = ivec3(-1, 1, 0);
    ivec3 t12 = ivec3(1, 0, 0); 
    ivec3 t13 = ivec3(0, 0, 0); 
    ivec3 t14 = ivec3(-1, 0, 0);
    ivec3 t15 = ivec3(1, -1, 0);
    ivec3 t16 = ivec3(0, -1, 0);
    ivec3 t17 = ivec3(-1, -1, 0);

    ivec3 t18 = ivec3(1, 1, 1); 
    ivec3 t19 = ivec3(0, 1, 1); 
    ivec3 t20 = ivec3(-1, 1, 1);
    ivec3 t21 = ivec3(1, 0, 1); 
    ivec3 t22 = ivec3(0, 0, 1); 
    ivec3 t23 = ivec3(-1, 0, 1);
    ivec3 t24 = ivec3(1, -1, 1);
    ivec3 t25 = ivec3(0, -1, 1);
    ivec3 t26 = ivec3(-1, -1, 1);

    float c1 = 0.015625f;
    float c2 = 0.03125f;
    float c4 = 0.0625f;
    float c8 = 0.125f;

    float ne_z_minus_1 = c1 * textureOffset(s, c, t00, 0).r;
    float n_z_minus_1 =  c2 * textureOffset(s, c, t01, 0).r;
    float nw_z_minus_1 = c1 * textureOffset(s, c, t02, 0).r;
    float e_z_minus_1 =  c2 * textureOffset(s, c, t03, 0).r;
    float c_z_minus_1 =  c4 * textureOffset(s, c, t04, 0).r;
    float w_z_minus_1 =  c2 * textureOffset(s, c, t05, 0).r;
    float se_z_minus_1 = c1 * textureOffset(s, c, t06, 0).r;
    float s_z_minus_1 =  c2 * textureOffset(s, c, t07, 0).r;
    float sw_z_minus_1 = c1 * textureOffset(s, c, t08, 0).r;
    float ne_z_0 =       c2 * textureOffset(s, c, t09, 0).r;
    float n_z_0 =        c4 * textureOffset(s, c, t10, 0).r;
    float nw_z_0 =       c2 * textureOffset(s, c, t11, 0).r;
    float e_z_0 =        c4 * textureOffset(s, c, t12, 0).r;
    float c_z_0 =        c8 * textureOffset(s, c, t13, 0).r;
    float w_z_0 =        c4 * textureOffset(s, c, t14, 0).r;
    float se_z_0 =       c2 * textureOffset(s, c, t15, 0).r;
    float s_z_0 =        c4 * textureOffset(s, c, t16, 0).r;
    float sw_z_0 =       c2 * textureOffset(s, c, t17, 0).r;
    float ne_z_plus_1 =  c1 * textureOffset(s, c, t18, 0).r;
    float n_z_plus_1 =   c2 * textureOffset(s, c, t19, 0).r;
    float nw_z_plus_1 =  c1 * textureOffset(s, c, t20, 0).r;
    float e_z_plus_1 =   c2 * textureOffset(s, c, t21, 0).r;
    float c_z_plus_1 =   c4 * textureOffset(s, c, t22, 0).r;
    float w_z_plus_1 =   c2 * textureOffset(s, c, t23, 0).r;
    float se_z_plus_1 =  c1 * textureOffset(s, c, t24, 0).r;
    float s_z_plus_1 =   c2 * textureOffset(s, c, t25, 0).r;
    float sw_z_plus_1 =  c1 * textureOffset(s, c, t26, 0).r;

    // The following boundary handling code will slightly improvement the
    // quality (average |r| goes down to 0.008 from 0.010) with the cost
    // of extra 0.5ms processing time every frame.
    //
    // Note that we simply substitute out-of-bound values with the boundary
    // values, which is not an accurate solution, but is so far satisfactory.
    // Also, our experiments reveal that without adjusting the coefficient of
    // boundary values to half(as per the restriction stencil), the algorithm
    // surprisingly gain a better result of average |r|. This could be, in my
    // opinion, a compensation to the highly unbalancing in the edge.

    ivec3 tex_size = textureSize(s, 0);
    if (f_coord.x >= tex_size.x - 1) {
        ne_z_minus_1 = n_z_minus_1;
        e_z_minus_1 =  c_z_minus_1;
        se_z_minus_1 = s_z_minus_1;
        ne_z_0 =       n_z_0;
        e_z_0 =        c_z_0;
        se_z_0 =       s_z_0;
        ne_z_plus_1 =  n_z_plus_1;
        e_z_plus_1 =   c_z_plus_1;
        se_z_plus_1 =  s_z_plus_1;
    }

    if (f_coord.x <= 0.5f) {
        nw_z_minus_1 = n_z_minus_1;
        w_z_minus_1 =  c_z_minus_1;
        sw_z_minus_1 = s_z_minus_1;
        nw_z_0 =       n_z_0;
        w_z_0 =        c_z_0;
        sw_z_0 =       s_z_0;
        nw_z_plus_1 =  n_z_plus_1;
        w_z_plus_1 =   c_z_plus_1;
        sw_z_plus_1 =  s_z_plus_1;
    }

    if (f_coord.z >= tex_size.z - 1) {
        ne_z_plus_1 = ne_z_0;
        n_z_plus_1 =  n_z_0;
        nw_z_plus_1 = nw_z_0;
        e_z_plus_1 =  e_z_0;
        c_z_plus_1 =  c_z_0;
        w_z_plus_1 =  w_z_0;
        se_z_plus_1 = se_z_0;
        s_z_plus_1 =  s_z_0;
        sw_z_plus_1 = sw_z_0;
    }

    if (f_coord.z <= 0.5f) {
        ne_z_minus_1 = ne_z_0;
        n_z_minus_1 =  n_z_0;
        nw_z_minus_1 = nw_z_0;
        e_z_minus_1 =  e_z_0;
        c_z_minus_1 =  c_z_0;
        w_z_minus_1 =  w_z_0;
        se_z_minus_1 = se_z_0;
        s_z_minus_1 =  s_z_0;
        sw_z_minus_1 = sw_z_0;
    }

    if (f_coord.y >= tex_size.y - 1) {
        ne_z_minus_1 = e_z_minus_1;
        n_z_minus_1 =  c_z_minus_1;
        nw_z_minus_1 = w_z_minus_1;
        ne_z_0 =       e_z_0;
        n_z_0 =        c_z_0;
        nw_z_0 =       w_z_0;
        ne_z_plus_1 =  e_z_plus_1;
        n_z_plus_1 =   c_z_plus_1;
        nw_z_plus_1 =  w_z_plus_1;
    }

    if (f_coord.y <= 0.5f) {
        se_z_minus_1 = e_z_minus_1;
        s_z_minus_1 =  c_z_minus_1;
        sw_z_minus_1 = w_z_minus_1;
        se_z_0 =       e_z_0;
        s_z_0 =        c_z_0;
        sw_z_0 =       w_z_0;
        se_z_plus_1 =  e_z_plus_1;
        s_z_plus_1 =   c_z_plus_1;
        sw_z_plus_1 =  w_z_plus_1;
    }

    float result =
        ne_z_minus_1 +
        n_z_minus_1 +
        nw_z_minus_1 +
        e_z_minus_1 +
        c_z_minus_1 +
        w_z_minus_1 +
        se_z_minus_1 +
        s_z_minus_1 +
        sw_z_minus_1 +
        
        ne_z_0 +
        n_z_0 +
        nw_z_0 +
        e_z_0 +
        c_z_0 +
        w_z_0 +
        se_z_0 +
        s_z_0 +
        sw_z_0 +
        
        ne_z_plus_1 +
        n_z_plus_1 +
        nw_z_plus_1 +
        e_z_plus_1 +
        c_z_plus_1 +
        w_z_plus_1 +
        se_z_plus_1 +
        s_z_plus_1 +
        sw_z_plus_1;
)";

const char* kProlongateCore = R"(
    // Accurate coordinates for accessing finer buffer is crucial here, since
    // we need exactly the original solution instead of an interpolated value.
    ivec3 f_coord = ivec3(gl_FragCoord.xy, gLayer);
    ivec3 c_coord = f_coord / 2;

    float c1 = 0.125f;
    float c2 = 0.25f;
    float c4 = 0.5f;
    float c8 = 1.0f;

    float d[4];
    d[0] = c8;
    d[1] = c4;
    d[2] = c2;
    d[3] = c1;

    int odd_x = f_coord.x - ((f_coord.x >> 1) << 1);
    int odd_y = f_coord.y - ((f_coord.y >> 1) << 1);
    int odd_z = f_coord.z - ((f_coord.z >> 1) << 1);

    int a0 = odd_x;
    int a1 = odd_y;
    int a2 = odd_z;
    int a3 = odd_x * odd_y;
    int a4 = odd_y * odd_z;
    int a5 = odd_x * odd_z;
    int a6 = odd_x * odd_y * odd_z;

    float u0 = a0 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 0)).r;
    float u1 = a1 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 0)).r;
    float u3 = a3 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(1, 1, 0)).r;
    float u2 = a2 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(0, 0, 1)).r;
    float u5 = a5 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 1)).r;
    float u4 = a4 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 1)).r;
    float u6 = a6 == 0 ? 0 : texelFetchOffset(c, c_coord, 0, ivec3(1, 1, 1)).r;

    float interpolated = texelFetch(c, c_coord, 0).r +
        u0 + u1 + u2 + u3 + u4 + u5 + u6;
    interpolated *= d[odd_x + odd_y + odd_z];
)";

const char* kResidualCore = R"(
    ivec3 coord = ivec3(gl_FragCoord.xy, gLayer);

    float near =   texelFetchOffset(packed_tex, coord, 0, ivec3(0, 0, -1)).r;
    float south =  texelFetchOffset(packed_tex, coord, 0, ivec3(0, -1, 0)).r;
    float west =   texelFetchOffset(packed_tex, coord, 0, ivec3(-1, 0, 0)).r;
    vec3 center =  texelFetch(packed_tex, coord, 0).rgb;
    float east =   texelFetchOffset(packed_tex, coord, 0, ivec3(1, 0, 0)).r;
    float north =  texelFetchOffset(packed_tex, coord, 0, ivec3(0, 1, 0)).r;
    float far =    texelFetchOffset(packed_tex, coord, 0, ivec3(0, 0, 1)).r;
    float b_center = center.g;

    ivec3 tex_size = textureSize(packed_tex, 0);
    if (coord.y == tex_size.y - 1)
        north = center.r;

    if (coord.y == 0)
        south = center.r;

    if (coord.x == tex_size.x - 1)
        east = center.r;

    if (coord.x == 0)
        west = center.r;

    if (coord.z == tex_size.z - 1)
        far = center.r;

    if (coord.z == 0)
        near = center.r;
)";
}

std::string MultigridShader::ComputeResidual()
{
    std::string part1 = R"(
out vec3 frag_color;

uniform sampler3D packed_tex;
uniform sampler3D b;

uniform float inverse_h_square;

in float gLayer;

void main()
{
)";
    std::string part2 = R"(
    b_center = texelFetch(b, coord, 0).r;
    float v = b_center -
        (north + south + east + west + far + near - 6.0 * center.r) *
            inverse_h_square;
    frag_color = vec3(v, 0.0f, 0.0f);
}
)";
    return part1 + kResidualCore + part2;
}

std::string MultigridShader::Restrict()
{
    std::string part1 = R"(
out vec3 frag_color;

uniform sampler3D s;
uniform vec3 inverse_size;

in float gLayer;

void main()
{
)";
    std::string part2 = R"(
    frag_color = vec3(result, 0.0f, 0.0f);
}
)";
    return part1 + kRestrictCore + part2;
}

std::string MultigridShader::Prolongate()
{
    std::string part1 = R"(
out vec3 frag_color;

uniform sampler3D fine;
uniform sampler3D c;

in float gLayer;

void main()
{
)";
    std::string part2 = R"(
    frag_color = vec3(texelFetch(fine, f_coord, 0).r + interpolated,
                      0.0f, 0.0f);
}
)";
    return part1 + kProlongateCore + part2;
}

std::string MultigridShader::RelaxWithZeroGuess()
{
    return R"(
out vec3 frag_color;

uniform sampler3D b;

uniform float alpha_omega_over_beta;

in float gLayer;

void main()
{
    ivec3 coord = ivec3(gl_FragCoord.xy, gLayer);
    frag_color = vec3(alpha_omega_over_beta * texelFetch(b, coord, 0).r,
                      0.0f, 0.0f);
}
)";
}

std::string MultigridShader::ComputeResidualPacked()
{
    std::string part1 = R"(
out vec3 frag_color;

uniform sampler3D packed_tex;

uniform float inverse_h_square;

in float gLayer;

void main()
{
)";
    std::string part2 = R"(
    float v = b_center -
        (north + south + east + west + far + near - 6.0 * center.r) *
            inverse_h_square;
    frag_color = vec3(center.r, b_center, v);
}
)";
    return part1 + kResidualCore + part2;
}

std::string MultigridShader::RelaxPacked()
{
    return R"(
out vec3 frag_color;

uniform sampler3D packed_tex;

uniform float one_minus_omega;
uniform float minus_h_square;
uniform float omega_over_beta;

in float gLayer;

void main()
{
    ivec3 coord = ivec3(gl_FragCoord.xy, gLayer);

    float near =   texelFetchOffset(packed_tex, coord, 0, ivec3(0, 0, -1)).r;
    float south =  texelFetchOffset(packed_tex, coord, 0, ivec3(0, -1, 0)).r;
    float west =   texelFetchOffset(packed_tex, coord, 0, ivec3(-1, 0, 0)).r;
    vec3 center =  texelFetch(packed_tex, coord, 0).rgb;
    float east =   texelFetchOffset(packed_tex, coord, 0, ivec3(1, 0, 0)).r;
    float north =  texelFetchOffset(packed_tex, coord, 0, ivec3(0, 1, 0)).r;
    float far =    texelFetchOffset(packed_tex, coord, 0, ivec3(0, 0, 1)).r;
    float b_center = center.g;

    ivec3 tex_size = textureSize(packed_tex, 0);
    if (coord.y == tex_size.y - 1)
        north = center.r;

    if (coord.y == 0)
        south = center.r;

    if (coord.x == tex_size.x - 1)
        east = center.r;

    if (coord.x == 0)
        west = center.r;

    if (coord.z == tex_size.z - 1)
        far = center.r;

    if (coord.z == 0)
        near = center.r;

    float v = one_minus_omega * center.r + 
        (north + south + east + west + far + near +
            minus_h_square * b_center) * omega_over_beta;

    frag_color = vec3(v, b_center, 0.0f);
}
)";
}

std::string MultigridShader::RelaxWithZeroGuessPacked()
{
    return R"(
out vec3 frag_color;

uniform sampler3D packed_tex;
uniform float alpha_omega_over_beta;
uniform float one_minus_omega;
uniform float minus_h_square;
uniform float omega_times_inverse_beta;

in float gLayer;

void main()
{
    ivec3 coord = ivec3(gl_FragCoord.xy, gLayer);

    float near =   texelFetchOffset(packed_tex, coord, 0, ivec3(0, 0, -1)).g;
    float south =  texelFetchOffset(packed_tex, coord, 0, ivec3(0, -1, 0)).g;
    float west =   texelFetchOffset(packed_tex, coord, 0, ivec3(-1, 0, 0)).g;
    vec3 center =  texelFetch(packed_tex, coord, 0).rgb;
    float east =   texelFetchOffset(packed_tex, coord, 0, ivec3(1, 0, 0)).g;
    float north =  texelFetchOffset(packed_tex, coord, 0, ivec3(0, 1, 0)).g;
    float far =    texelFetchOffset(packed_tex, coord, 0, ivec3(0, 0, 1)).g;
    float b_center = center.g;

    ivec3 tex_size = textureSize(packed_tex, 0);
    if (coord.y == tex_size.y - 1)
        north = b_center;

    if (coord.y == 0)
        south = b_center;

    if (coord.x == tex_size.x - 1)
        east = b_center;

    if (coord.x == 0)
        west = b_center;

    if (coord.z == tex_size.z - 1)
        far = b_center;

    if (coord.z == 0)
        near = b_center;

    float v = one_minus_omega * (alpha_omega_over_beta * b_center) + 
        (alpha_omega_over_beta * (north + south + east + west + far + near) +
            minus_h_square * b_center) * omega_times_inverse_beta;

    frag_color = vec3(v, b_center, 0.0f);
}
)";
}

std::string MultigridShader::ComputeResidualPackedDiagnosis()
{
    std::string part1 = R"(
out vec3 frag_color;

uniform sampler3D packed_tex;

uniform float inverse_h_square;

in float gLayer;

void main()
{
)";
    std::string part2 = R"(
    float v = b_center -
        (north + south + east + west + far + near - 6.0 * center.r) *
            inverse_h_square;
    frag_color = vec3(abs(v), 0.0f, 0.0f);
}
)";
    return part1 + kResidualCore + part2;
}
