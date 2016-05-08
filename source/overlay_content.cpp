#include "stdafx.h"
#include "overlay_content.h"

#include "shader/overlay_shader.h"
#include "utility.h"

OverlayContent::OverlayContent()
    : last_text_()
    , texture_(0)
    , width_(250)
    , height_(200)
    , program_(0)
    , quad_mesh_(nullptr)
    , bitmap_buf_()
{
}

OverlayContent::~OverlayContent()
{
    if (bitmap_buf_.hbitmap_) {
        DeleteObject(bitmap_buf_.hbitmap_);
        bitmap_buf_.hbitmap_ = nullptr;
        bitmap_buf_.bits_ = nullptr;
    }

    if (quad_mesh_) {
        delete quad_mesh_;
        quad_mesh_ = nullptr;
    }
}

void OverlayContent::RenderText(const std::string& text, int viewport_width,
                                int viewport_height)
{
    if (last_text_ == text)
        return;

    HDC hdc = ::CreateCompatibleDC(nullptr);
    if (!hdc)
        return;
    
    BitmapBuf bitmap_buf = GetBitmapBuf();
    if (!bitmap_buf.hbitmap_)
        return;

    RECT bounds = {0, 0, width_, height_};
    HGDIOBJ old_bitmap = ::SelectObject(hdc, bitmap_buf.hbitmap_);
    ::SetBkMode(hdc, TRANSPARENT);
    ::SetTextColor(hdc, RGB(255, 255, 255));
    ::DrawTextA(hdc, text.c_str(), text.length(), &bounds, DT_LEFT);

    ::SelectObject(hdc, old_bitmap);
    DeleteDC(hdc);

    glBindTexture(GL_TEXTURE_2D, GetTexture());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width_, height_, 0, GL_RG,
                 GL_UNSIGNED_BYTE, bitmap_buf.bits_);

    glEnable(GL_BLEND);

    glUseProgram(GetProgram());
    SetUniform("depth", 1.0f);
    SetUniform("sampler", 0);
    SetUniform("viewport_size", static_cast<float>(viewport_width),
               static_cast<float>(viewport_height));

    glBindTexture(GL_TEXTURE_2D, GetTexture());
    RenderMesh(*GetQuadMesh());

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
    glUseProgram(0);
}

GLuint OverlayContent::GetTexture()
{
    if (!texture_) {
        glGenTextures(1, &(texture_));
    }

    return texture_;
}

GLuint OverlayContent::GetProgram()
{
    if (!program_) {
        program_ = LoadProgram(OverlayShader::Vertex(), "",
                               OverlayShader::Fragment());
    }

    return program_;
}

MeshPod* OverlayContent::GetQuadMesh()
{
    if (!quad_mesh_) {
        quad_mesh_ = new MeshPod(CreateQuadMesh(-1.0f, 1.0f, 1.0f, -1.0f));
    }

    return quad_mesh_;
}

OverlayContent::BitmapBuf OverlayContent::GetBitmapBuf()
{
    HDC hdc = ::CreateCompatibleDC(nullptr);
    if (!hdc)
        return BitmapBuf();

    RECT bounds = {0, 0, width_, height_};

    BITMAPINFOHEADER bitmap_header = {};
    bitmap_header.biSize = sizeof(bitmap_header);
    bitmap_header.biBitCount = 16;
    bitmap_header.biCompression = BI_RGB;
    bitmap_header.biPlanes = 1;
    bitmap_header.biWidth = bounds.right - bounds.left;
    bitmap_header.biHeight = -bounds.top - bounds.bottom;

    bitmap_buf_.hbitmap_ = ::CreateDIBSection(
        hdc, reinterpret_cast<BITMAPINFO*>(&bitmap_header), DIB_RGB_COLORS,
        reinterpret_cast<void**>(&bitmap_buf_.bits_), nullptr, 0);

    DeleteDC(hdc);
    return bitmap_buf_;
}
