#include "stdafx.h"
#include "overlay_content.h"

#include "overlay_shader.h"
#include "Utility.h"

OverlayContent::OverlayContent()
    : last_text_()
    , texture_(0)
    , width_(100)
    , height_(50)
    , program_(0)
    , quad_mesh_(nullptr)
{
}

OverlayContent::~OverlayContent()
{
    if (quad_mesh_) {
        delete quad_mesh_;
        quad_mesh_ = nullptr;
    }
}

void OverlayContent::RenderText(const std::string& text)
{
    if (last_text_ == text)
        return;

    //last_text_ = text;

    // Draw text into a GDI DC.
    HDC hdc = ::CreateCompatibleDC(nullptr);
    if (!hdc)
        return;

    RECT bounds = {0, 0, width_, height_};

    BITMAPINFOHEADER bitmap_header = {};
    bitmap_header.biSize = sizeof(bitmap_header);
    bitmap_header.biBitCount = 16;
    bitmap_header.biCompression = BI_RGB;
    bitmap_header.biPlanes = 1;
    bitmap_header.biWidth = bounds.right - bounds.left;
    bitmap_header.biHeight = -bounds.top - bounds.bottom;

    char* bits = nullptr;
    HBITMAP hbitmap = ::CreateDIBSection(
        hdc, reinterpret_cast<BITMAPINFO*>(&bitmap_header), DIB_RGB_COLORS,
        reinterpret_cast<void**>(&bits), nullptr, 0);
    if (!hbitmap) {
        DeleteDC(hdc);
        return;
    }

    HGDIOBJ old_bitmap = ::SelectObject(hdc, hbitmap);
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
                 GL_UNSIGNED_BYTE, bits);

    DeleteObject(hbitmap);

    glEnable(GL_BLEND);
    glUseProgram(GetProgram());
    SetUniform("depth", 1.0f);
    SetUniform("sampler", 0);
    SetUniform("viewport_size", static_cast<float>(ViewportWidth),
               static_cast<float>(ViewportWidth));
    glActiveTexture(GL_TEXTURE0);
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
        program_ = LoadProgram(OverlayShader::GetVertexShaderCode(), "",
                               OverlayShader::GetFragmentShaderCode());
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