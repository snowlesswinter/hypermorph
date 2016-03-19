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

    // Pass 1: 2-center cells.
    if (f_coord.x > tex_size.x) {
        center_east_center = center_center_center;
    }

    if (f_coord.x < 1.01f) { // Please note that it is 1.0. We are in
                             // staggered grid, and be careful about the
                             // shader coordinates.
        center_west_center = center_center_center;
    }

    if (f_coord.z > tex_size.z) {
        center_center_far = center_center_center;
    }

    if (f_coord.z < 1.01f) {
        center_center_near = center_center_center;
    }

    if (f_coord.y > tex_size.y) {
        north_center_center = center_center_center;
    }

    if (f_coord.y < 1.01f) {
        south_center_center = center_center_center;
    }

    // Pass 2: 1-center cells.
    if (f_coord.x > tex_size.x) {
        center_east_near =  center_center_near;
        north_east_center = north_center_center;
        south_east_center = south_center_center;
        center_east_far =   center_center_far;
    }

    if (f_coord.x < 1.01f) {
        center_west_near =  center_center_near;
        north_west_center = north_center_center;
        south_west_center = south_center_center;
        center_west_far =   center_center_far;
    }

    if (f_coord.z > tex_size.z) {
        north_center_far = north_center_center;
        center_east_far =  center_east_center;
        center_west_far =  center_west_center;
        south_center_far = south_center_center;
    }

    if (f_coord.z < 1.01f) {
        north_center_near = north_center_center;
        center_east_near =  center_east_center;
        center_west_near =  center_west_center;
        south_center_near = south_center_center;
    }

    if (f_coord.y > tex_size.y) {
        north_center_near = center_center_near;
        north_east_center = center_east_center;
        north_west_center = center_west_center;
        north_center_far =  center_center_far;
    }

    if (f_coord.y < 1.01f) {
        south_center_near = center_center_near;
        south_east_center = center_east_center;
        south_west_center = center_west_center;
        south_center_far =  center_center_far;
    }

    // Pass 3: corner cells.
    if (f_coord.x > tex_size.x) {
        north_east_near = north_center_near;
        south_east_near = south_center_near;
        north_east_far =  north_center_far;
        south_east_far =  south_center_far;
    }

    if (f_coord.x < 1.01f) {
        north_west_near = north_center_near;
        south_west_near = south_center_near;
        north_west_far =  north_center_far;
        south_west_far =  south_center_far;
    }

    if (f_coord.z > tex_size.z) {
        north_east_far = north_east_center;
        north_west_far = north_west_center;
        south_east_far = south_east_center;
        south_west_far = south_west_center;
    }

    if (f_coord.z < 1.01f) {
        north_east_near = north_east_center;
        north_west_near = north_west_center;
        south_east_near = south_east_center;
        south_west_near = south_west_center;
    }

    if (f_coord.y > tex_size.y) {
        north_east_near = center_east_near;
        north_west_near = center_west_near;
        north_east_far =  center_east_far;
        north_west_far =  center_west_far;
    }

    if (f_coord.y < 1.01f) {
        south_east_near = center_east_near;
        south_west_near = center_west_near;
        south_east_far =  center_east_far;
        south_west_far =  center_west_far;
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
} // Anonymous namespace

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
