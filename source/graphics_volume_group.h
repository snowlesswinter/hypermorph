#ifndef _GRAPHICS_VOLUME_GROUP_H_
#define _GRAPHICS_VOLUME_GROUP_H_

#include <array>
#include <memory>

#include "graphics_volume.h"

class GraphicsVolume3
{
private:
    typedef std::shared_ptr<GraphicsVolume> GraphicsVolume3::* BoolType;

public:
    static int num_of_volumes() { return num_of_volumes_; }

    explicit GraphicsVolume3(GraphicsLib graphics_lib);
    GraphicsVolume3(GraphicsVolume3&& obj);
    GraphicsVolume3(const std::shared_ptr<GraphicsVolume>& x,
                    const std::shared_ptr<GraphicsVolume>& y,
                    const std::shared_ptr<GraphicsVolume>& z);

    bool Assign(const std::shared_ptr<GraphicsVolume>& x,
                const std::shared_ptr<GraphicsVolume>& y,
                const std::shared_ptr<GraphicsVolume>& z);
    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width, int border);
    void Swap(GraphicsVolume3& obj);
    void Swap(std::shared_ptr<GraphicsVolume>* x,
              std::shared_ptr<GraphicsVolume>* y,
              std::shared_ptr<GraphicsVolume>* z);

    inline const std::shared_ptr<GraphicsVolume>& x() const { return x_; }
    inline const std::shared_ptr<GraphicsVolume>& y() const { return y_; }
    inline const std::shared_ptr<GraphicsVolume>& z() const { return z_; }

    const std::shared_ptr<GraphicsVolume>& operator[](int n) const;
    operator BoolType() const;

private:
    static const int num_of_volumes_ = 3;

    GraphicsVolume3(const GraphicsVolume3&);
    GraphicsVolume3& operator=(const GraphicsVolume3&);

    GraphicsLib graphics_lib_;
    std::shared_ptr<GraphicsVolume> x_;
    std::shared_ptr<GraphicsVolume> y_;
    std::shared_ptr<GraphicsVolume> z_;
    std::array<std::shared_ptr<GraphicsVolume>*, num_of_volumes_> v_;
};

#endif // _GRAPHICS_VOLUME_GROUP_H_