#include "stdafx.h"
#include "multigrid_shader.h"

std::string MultigridShader::GetComputeResidualShaderCode()
{
    return R"(
out vec3 frag_color;

uniform sampler3D residual;
uniform sampler3D u;
uniform sampler3D b;

uniform float inverse_h_square;

in float gLayer;

void main()
{
    ivec3 coord = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring pressure:
    float pN = texelFetchOffset(u, coord, 0, ivec3(0, 1, 0)).r;
    float pS = texelFetchOffset(u, coord, 0, ivec3(0, -1, 0)).r;
    float pE = texelFetchOffset(u, coord, 0, ivec3(1, 0, 0)).r;
    float pW = texelFetchOffset(u, coord, 0, ivec3(-1, 0, 0)).r;
    float pU = texelFetchOffset(u, coord, 0, ivec3(0, 0, 1)).r;
    float pD = texelFetchOffset(u, coord, 0, ivec3(0, 0, -1)).r;
    float pC = texelFetch(u, coord, 0).r;

    // Handle boundary problem
    ivec3 tex_size = textureSize(u, 0);
    if (coord.y >= tex_size.y - 1)
        pN = pC;

    if (coord.y <= 0)
        pS = pC;

    if (coord.x >= tex_size.x - 1)
        pE = pC;

    if (coord.x <= 0)
        pW = pC;

    if (coord.z >= tex_size.z - 1)
        pU = pC;

    if (coord.z <= 0)
        pD = pC;

    float bC = texelFetch(b, coord, 0).r;
    frag_color = vec3(
        bC - (pW + pE + pS + pN + pU + pD - 6.0 * pC) * inverse_h_square,
        0.0f, 0.0f);
}
)";
}

std::string MultigridShader::GetRestrictShaderCode()
{
    return R"(
out vec3 frag_color;

uniform sampler3D s;

in float gLayer;

void main()
{
    ivec3 c = ivec3(gl_FragCoord.xy, gLayer) * 2;

    float c1 = 0.015625f;
    float c2 = 0.03125f;
    float c4 = 0.0625f;
    float c8 = 0.125f;

    float ne_z_minus_1 = c1 * texelFetchOffset(s, c, 0, ivec3(1, 1, -1)).r;
    float n_z_minus_1  = c2 * texelFetchOffset(s, c, 0, ivec3(0, 1, -1)).r;
    float nw_z_minus_1 = c1 * texelFetchOffset(s, c, 0, ivec3(-1, 1, -1)).r;
    float e_z_minus_1 =  c2 * texelFetchOffset(s, c, 0, ivec3(1, 0, -1)).r;
    float c_z_minus_1 =  c4 * texelFetchOffset(s, c, 0, ivec3(0, 0, -1)).r;
    float w_z_minus_1 =  c2 * texelFetchOffset(s, c, 0, ivec3(-1, 0, -1)).r;
    float se_z_minus_1 = c1 * texelFetchOffset(s, c, 0, ivec3(1, -1, -1)).r;
    float s_z_minus_1 =  c2 * texelFetchOffset(s, c, 0, ivec3(0, -1, -1)).r;
    float sw_z_minus_1 = c1 * texelFetchOffset(s, c, 0, ivec3(-1, -1, -1)).r;

    float ne_z_0 =       c2 * texelFetchOffset(s, c, 0, ivec3(1, 1, 0)).r;
    float n_z_0  =       c4 * texelFetchOffset(s, c, 0, ivec3(0, 1, 0)).r;
    float nw_z_0 =       c2 * texelFetchOffset(s, c, 0, ivec3(-1, 1, 0)).r;
    float e_z_0 =        c4 * texelFetchOffset(s, c, 0, ivec3(1, 0, 0)).r;
    float c_z_0 =        c8 * texelFetch(s, c, 0).r;
    float w_z_0 =        c4 * texelFetchOffset(s, c, 0, ivec3(-1, 0, 0)).r;
    float se_z_0 =       c2 * texelFetchOffset(s, c, 0, ivec3(1, -1, 0)).r;
    float s_z_0 =        c4 * texelFetchOffset(s, c, 0, ivec3(0, -1, 0)).r;
    float sw_z_0 =       c2 * texelFetchOffset(s, c, 0, ivec3(-1, -1, 0)).r;

    float ne_z_plus_1 =  c1 * texelFetchOffset(s, c, 0, ivec3(1, 1, 1)).r;
    float n_z_plus_1  =  c2 * texelFetchOffset(s, c, 0, ivec3(0, 1, 1)).r;
    float nw_z_plus_1 =  c1 * texelFetchOffset(s, c, 0, ivec3(-1, 1, 1)).r;
    float e_z_plus_1 =   c2 * texelFetchOffset(s, c, 0, ivec3(1, 0, 1)).r;
    float c_z_plus_1 =   c4 * texelFetchOffset(s, c, 0, ivec3(0, 0, 1)).r;
    float w_z_plus_1 =   c2 * texelFetchOffset(s, c, 0, ivec3(-1, 0, 1)).r;
    float se_z_plus_1 =  c1 * texelFetchOffset(s, c, 0, ivec3(1, -1, 1)).r;
    float s_z_plus_1 =   c2 * texelFetchOffset(s, c, 0, ivec3(0, -1, 1)).r;
    float sw_z_plus_1 =  c1 * texelFetchOffset(s, c, 0, ivec3(-1, -1, 1)).r;

//     ivec3 tex_size = textureSize(pressure, 0);
//     if (c.y >= tex_size.y - 1)
//         pN = pC;
// 
//     if (c.y <= 0)
//         pS = pC;
// 
//     if (c.x >= tex_size.x - 1)
//         pE = pC;
// 
//     if (c.x <= 0)
//         pW = pC;
// 
//     if (c.z >= tex_size.z - 1)
//         pU = pC;
// 
//     if (c.z <= 0)
//         pD = pC;

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

    frag_color = vec3(result, 0.0f, 0.0f);
}
)";
}

std::string MultigridShader::GetProlongateShaderCode()
{
    return R"(
out vec3 frag_color;

uniform sampler3D fine;
uniform sampler3D c;

in float gLayer;

void main()
{
    // Accurate coordinates for accessing finer buffer is crucial here, since
    // we need exactly the original solution instead of an interpolated value.
    ivec3 f_coord = ivec3(gl_FragCoord.xy, gLayer);
    ivec3 c_coord = f_coord / 2;

    float c1 = 0.125f;
    float c2 = 0.25f;
    float c4 = 0.5f;
    float c8 = 1.0f;

    float interpolated = 0.0f;
    if (f_coord.x % 2 == 0) {
        if (f_coord.y % 2 == 0) {
            if (f_coord.z % 2 == 0) {
                interpolated = texture(c, c_coord).r; // * c8
            } else {
                interpolated = c4 *
                    (texelFetch(c, c_coord, 0).r +
                        texelFetchOffset(c, c_coord, 0, ivec3(0, 0, 1)).r);
            }
        } else {
            if (f_coord.z % 2 == 0) {
                interpolated = c4 *
                    (texelFetch(c, c_coord, 0).r +
                        texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 0)).r);
            } else {
                interpolated = c2 *
                    (texelFetch(c, c_coord, 0).r +
                        texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 0)).r +
                        texelFetchOffset(c, c_coord, 0, ivec3(0, 0, 1)).r + 
                        texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 1)).r);
            }
        }
    } else if (f_coord.y % 2 == 0) {
        if (f_coord.z % 2 == 0) {
            interpolated = c4 *
                (texelFetch(c, c_coord, 0).r +
                    texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 0)).r);
        } else {
            interpolated = c2 *
                (texelFetch(c, c_coord, 0).r +
                    texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 0)).r +
                    texelFetchOffset(c, c_coord, 0, ivec3(0, 0, 1)).r + 
                    texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 1)).r);
        }
    } else if (f_coord.z % 2 == 0) {
        interpolated = c2 *
            (texelFetch(c, c_coord, 0).r +
                texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 0)).r +
                texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 0)).r + 
                texelFetchOffset(c, c_coord, 0, ivec3(1, 1, 0)).r);
    } else {
        interpolated = c1 *
            (texelFetch(c, c_coord, 0).r +
                texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 0)).r +
                texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 0)).r + 
                texelFetchOffset(c, c_coord, 0, ivec3(0, 0, 1)).r + 
                texelFetchOffset(c, c_coord, 0, ivec3(1, 1, 0)).r + 
                texelFetchOffset(c, c_coord, 0, ivec3(0, 1, 1)).r +
                texelFetchOffset(c, c_coord, 0, ivec3(1, 0, 1)).r + 
                texelFetchOffset(c, c_coord, 0, ivec3(1, 1, 1)).r);
    }

    frag_color = vec3(texelFetch(fine, f_coord, 0).r + interpolated,
                      0.0f, 0.0f);
}
)";
}

std::string MultigridShader::GetAbsoluteShaderCode()
{
    return R"(
out vec3 frag_color;

uniform sampler3D t;

in float gLayer;

void main()
{
    ivec3 coord = ivec3(gl_FragCoord.xy, gLayer);
    frag_color = vec3(abs(texelFetch(t, coord, 0).r), 0.0f, 0.0f);
}
)";
}
