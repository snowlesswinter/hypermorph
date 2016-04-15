#include "stdafx.h"
#include "overlay_shader.h"

std::string OverlayShader::Vertex()
{
    return R"(
in vec4 position;
in vec2 tex_coord;
out vec2 v_coord;

uniform float depth;

void main()
{
    gl_Position = position;
    gl_Position.z = depth;
    v_coord = tex_coord;
}
)";
}

std::string OverlayShader::Fragment()
{
    return R"(
in vec2 v_coord;
out vec4 frag_color;

uniform sampler2D sampler;
uniform vec2 viewport_size;

void main()
{
    ivec2 tex_size = textureSize(sampler, 0);
    ivec2 coord = ivec2(v_coord.x * viewport_size.x,
                        v_coord.y * viewport_size.y);
    if (coord.x >= tex_size.x || coord.y >= tex_size.y) {
        discard;
    }

    vec4 c = texelFetch(sampler, coord, 0);
    frag_color = vec4(c.r, c.r, 0, c.r);
}
)";
}