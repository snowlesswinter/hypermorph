#include "stdafx.h"
#include "multigrid_staggered_shader.h"

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

    float north_east_near =      c1 * textureOffset(s, c, t00, 0).r;
    float north_center_near =    c2 * textureOffset(s, c, t01, 0).r;
    float north_west_near =      c1 * textureOffset(s, c, t02, 0).r;
    float center_east_near =     c2 * textureOffset(s, c, t03, 0).r;
    float center_center_near =   c4 * textureOffset(s, c, t04, 0).r;
    float center_west_near =     c2 * textureOffset(s, c, t05, 0).r;
    float south_east_near =      c1 * textureOffset(s, c, t06, 0).r;
    float south_center_near =    c2 * textureOffset(s, c, t07, 0).r;
    float south_west_near =      c1 * textureOffset(s, c, t08, 0).r;
    float north_east_center =    c2 * textureOffset(s, c, t09, 0).r;
    float north_center_center =  c4 * textureOffset(s, c, t10, 0).r;
    float north_west_center =    c2 * textureOffset(s, c, t11, 0).r;
    float center_east_center =   c4 * textureOffset(s, c, t12, 0).r;
    float center_center_center = c8 * textureOffset(s, c, t13, 0).r;
    float center_west_center =   c4 * textureOffset(s, c, t14, 0).r;
    float south_east_center =    c2 * textureOffset(s, c, t15, 0).r;
    float south_center_center =  c4 * textureOffset(s, c, t16, 0).r;
    float south_west_center =    c2 * textureOffset(s, c, t17, 0).r;
    float north_east_far =       c1 * textureOffset(s, c, t18, 0).r;
    float north_center_far =     c2 * textureOffset(s, c, t19, 0).r;
    float north_west_far =       c1 * textureOffset(s, c, t20, 0).r;
    float center_east_far =      c2 * textureOffset(s, c, t21, 0).r;
    float center_center_far =    c4 * textureOffset(s, c, t22, 0).r;
    float center_west_far =      c2 * textureOffset(s, c, t23, 0).r;
    float south_east_far =       c1 * textureOffset(s, c, t24, 0).r;
    float south_center_far =     c2 * textureOffset(s, c, t25, 0).r;
    float south_west_far =       c1 * textureOffset(s, c, t26, 0).r;

    // The following boundary handling code will significantly improvement the
    // quality(most apparently the max |r|), especially for the cases that
    // high divergence piling up at the boundaries. And most surprising, the
    // extra branching code path doesn't stall the execution, but make the
    // whole solving procedure a few milliseconds faster.

    ivec3 size_int = textureSize(s, 0);
    vec3 tex_size = vec3(float(size_int.x) - 1.01f,
                         float(size_int.y) - 1.01f,
                         float(size_int.z) - 1.01f);

    // We have to deal with all of the boundaries, including faces, edges, and
    // corners. To avoid too much branch within the shader, we played a trick,
    // that we split the whole process into 3 pass, with 2, 1, 0 "center cells"
    // respectively, so that if the cell locates on the edge or at the corner,
    // we don't have to write tons of if statement, and just let the code path
    // "fall back" to a previous rectification.
    //
    // The mysterious thing is, our experiments revealed that without
    // adjusting the stencil of boundary values(they are supposed to be
    // multiplied 0.5), the algorithm surprisingly had gained a better result
    // of avg/max |r|. This could be, in my guess, a compensation to the
    // highly unbalancing near the boundary.
    //
    // Update: After rewriting the prolongation algorithm, I found some jitter
    //         near the corner, immediately I recalled this factor, so I
    //         changed it back to 0.5f as expected in the first place, and the
    //         jitter disappeared as I wished! The code finally makes sense
    //         again. But the bad news is, the avg/max |r| had slightly arisen.

    float scale = 0.5f;

    // Pass 1: 2-center cells.
    if (f_coord.x > tex_size.x) {
        center_east_center = center_center_center;
    }

    if (f_coord.x < 1.01f) { // Please note that it is 1.0. We are in
                             // staggered grid, and be careful about the
                             // shader coordinates.
        center_west_center = scale * center_center_center;
    }

    if (f_coord.z > tex_size.z) {
        center_center_far = scale * center_center_center;
    }

    if (f_coord.z < 1.01f) {
        center_center_near = scale * center_center_center;
    }

    if (f_coord.y > tex_size.y) {
        north_center_center = scale * center_center_center;
    }

    if (f_coord.y < 1.01f) {
        south_center_center = scale * center_center_center;
    }

    // Pass 2: 1-center cells.
    if (f_coord.x > tex_size.x) {
        center_east_near =  scale * center_center_near;
        north_east_center = scale * north_center_center;
        south_east_center = scale * south_center_center;
        center_east_far =   scale * center_center_far;
    }

    if (f_coord.x < 1.01f) {
        center_west_near =  scale * center_center_near;
        north_west_center = scale * north_center_center;
        south_west_center = scale * south_center_center;
        center_west_far =   scale * center_center_far;
    }

    if (f_coord.z > tex_size.z) {
        north_center_far = scale * north_center_center;
        center_east_far =  scale * center_east_center;
        center_west_far =  scale * center_west_center;
        south_center_far = scale * south_center_center;
    }

    if (f_coord.z < 1.01f) {
        north_center_near = scale * north_center_center;
        center_east_near =  scale * center_east_center;
        center_west_near =  scale * center_west_center;
        south_center_near = scale * south_center_center;
    }

    if (f_coord.y > tex_size.y) {
        north_center_near = scale * center_center_near;
        north_east_center = scale * center_east_center;
        north_west_center = scale * center_west_center;
        north_center_far =  scale * center_center_far;
    }

    if (f_coord.y < 1.01f) {
        south_center_near = scale * center_center_near;
        south_east_center = scale * center_east_center;
        south_west_center = scale * center_west_center;
        south_center_far =  scale * center_center_far;
    }

    // Pass 3: corner cells.
    if (f_coord.x > tex_size.x) {
        north_east_near = scale * north_center_near;
        south_east_near = scale * south_center_near;
        north_east_far =  scale * north_center_far;
        south_east_far =  scale * south_center_far;
    }

    if (f_coord.x < 1.01f) {
        north_west_near = scale * north_center_near;
        south_west_near = scale * south_center_near;
        north_west_far =  scale * north_center_far;
        south_west_far =  scale * south_center_far;
    }

    if (f_coord.z > tex_size.z) {
        north_east_far = scale * north_east_center;
        north_west_far = scale * north_west_center;
        south_east_far = scale * south_east_center;
        south_west_far = scale * south_west_center;
    }

    if (f_coord.z < 1.01f) {
        north_east_near = scale * north_east_center;
        north_west_near = scale * north_west_center;
        south_east_near = scale * south_east_center;
        south_west_near = scale * south_west_center;
    }

    if (f_coord.y > tex_size.y) {
        north_east_near = scale * center_east_near;
        north_west_near = scale * center_west_near;
        north_east_far =  scale * center_east_far;
        north_west_far =  scale * center_west_far;
    }

    if (f_coord.y < 1.01f) {
        south_east_near = scale * center_east_near;
        south_west_near = scale * center_west_near;
        south_east_far =  scale * center_east_far;
        south_west_far =  scale * center_west_far;
    }

    float result =
        north_east_near +
        north_center_near +
        north_west_near +
        center_east_near +
        center_center_near +
        center_west_near +
        south_east_near +
        south_center_near +
        south_west_near +
        
        north_east_center +
        north_center_center +
        north_west_center +
        center_west_center +
        center_center_center +
        center_west_center +
        south_east_center +
        south_center_center +
        south_west_center +
        
        north_east_far +
        north_center_far +
        north_west_far +
        center_east_far +
        center_center_far +
        center_west_far +
        south_east_far +
        south_center_far +
        south_west_far;
)";

const char* kProlongateCoreBackup = R"(
    vec3 f_coord =      vec3(gl_FragCoord.xy, gLayer);
    ivec3 f_coord_int = ivec3(f_coord);
    vec3 c_coord =      inverse_size * (vec3(f_coord_int) * 0.5f);

    float c1 = 0.125f;
    float c2 = 0.25f;
    float c4 = 0.5f;
    float c8 = 1.0f;

    float d[4];
    d[0] = c8;
    d[1] = c4;
    d[2] = c2;
    d[3] = c1;

    int odd_x = f_coord_int.x - ((f_coord_int.x >> 1) << 1);
    int odd_y = f_coord_int.y - ((f_coord_int.y >> 1) << 1);
    int odd_z = f_coord_int.z - ((f_coord_int.z >> 1) << 1);

    int a0 = odd_x;
    int a1 = odd_y;
    int a2 = odd_z;
    int a3 = odd_x * odd_y;
    int a4 = odd_y * odd_z;
    int a5 = odd_x * odd_z;
    int a6 = odd_x * odd_y * odd_z;

    float u0 = a0 == 0 ? 0 : textureOffset(s, c_coord, ivec3(1, 0, 0), 0).r;
    float u1 = a1 == 0 ? 0 : textureOffset(s, c_coord, ivec3(0, 1, 0), 0).r;
    float u3 = a3 == 0 ? 0 : textureOffset(s, c_coord, ivec3(1, 1, 0), 0).r;
    float u2 = a2 == 0 ? 0 : textureOffset(s, c_coord, ivec3(0, 0, 1), 0).r;
    float u5 = a5 == 0 ? 0 : textureOffset(s, c_coord, ivec3(1, 0, 1), 0).r;
    float u4 = a4 == 0 ? 0 : textureOffset(s, c_coord, ivec3(0, 1, 1), 0).r;
    float u6 = a6 == 0 ? 0 : textureOffset(s, c_coord, ivec3(1, 1, 1), 0).r;

    float interpolated = texture(s, c_coord, 0).r +
        u0 + u1 + u2 + u3 + u4 + u5 + u6;
    interpolated *= d[odd_x + odd_y + odd_z];
)";

const char* kProlongateCore = R"(
    vec3 f_coord =      vec3(gl_FragCoord.xy, gLayer);
    ivec3 f_coord_int = ivec3(f_coord);
    vec3 c =            vec3(f_coord_int) * 0.5f;

    int odd_x = f_coord_int.x - ((f_coord_int.x >> 1) << 1);
    int odd_y = f_coord_int.y - ((f_coord_int.y >> 1) << 1);
    int odd_z = f_coord_int.z - ((f_coord_int.z >> 1) << 1);

    float t_x = -1.0f * (1 - odd_x) * 0.08333333f;
    float t_y = -1.0f * (1 - odd_y) * 0.08333333f;
    float t_z = -1.0f * (1 - odd_z) * 0.08333333f;

    vec3 t_c = c + vec3(t_x, t_y, t_z);
    float result = texture(s, inverse_size_c * t_c, 0).r;
)";

} // Anonymous namespace

std::string MultigridStaggeredShader::ProlongatePacked()
{
    std::string part1 = R"(
out vec3 frag_color;

uniform sampler3D fine;
uniform sampler3D s;
uniform vec3 inverse_size_f;
uniform vec3 inverse_size_c;

in float gLayer;

void main()
{
)";
    std::string part2 = R"(
    vec3 f = texture(fine, inverse_size_f * f_coord, 0).rgb;
    frag_color = vec3(f.r + result, f.g, 0.0f);
}
)";
    return part1 + kProlongateCore + part2;
}

std::string MultigridStaggeredShader::RestrictResidualPacked()
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

    frag_color = vec3(0.0f, result, 0.0f);
}
)";
    std::string restrict_core = kRestrictCore;
    std::regex e("\\)\\.r;");
    std::string core = std::regex_replace(restrict_core, e, ").b;");
    return part1 + core + part2;
}

std::string MultigridStaggeredShader::RestrictPacked()
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

    frag_color = vec3(result.r, result.g, 0.0f);
}
)";
    std::string restrict_core = kRestrictCore;
    std::regex e1("\\)\\.r;");
    std::string core = std::regex_replace(restrict_core, e1, ").rgb;");
    std::regex e2("float\\s([a-z]{5,6}_)");
    restrict_core = std::regex_replace(core, e2, "vec3 $1");
    std::regex e3("float result");
    core = std::regex_replace(restrict_core, e3, "vec3 result");
    return part1 + core + part2;
}
