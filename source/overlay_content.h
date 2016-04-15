#ifndef _OVERLAY_CONTENT_H_
#define _OVERLAY_CONTENT_H_

#include <string>

#include "third_party/opengl/glew.h"

struct MeshPod;
class OverlayContent
{
public:
    OverlayContent();
    ~OverlayContent();

    void RenderText(const std::string& text);

private:
    struct BitmapBuf
    {
        HBITMAP hbitmap_;
        void* bits_;

        BitmapBuf() : hbitmap_(nullptr), bits_(nullptr) {}
    };
    GLuint GetTexture();
    GLuint GetProgram();
    MeshPod* GetQuadMesh();
    BitmapBuf GetBitmapBuf();

    std::string last_text_;
    GLuint texture_ = 0;
    GLsizei width_ = 0;
    GLsizei height_ = 0;
    GLuint program_ = 0;
    MeshPod* quad_mesh_;
    BitmapBuf bitmap_buf_;
};

#endif // _OVERLAY_CONTENT_H_