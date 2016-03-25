#ifndef _GRAPHICS_VOLUME_H_
#define _GRAPHICS_VOLUME_H_

#include <map>
#include <memory>

class GraphicsVolume
{
public:
    GraphicsVolume();
    ~GraphicsVolume();

    virtual bool Create(int width, int height, int depth, int num_of_components,
                        int byte_width) = 0;

};

#endif // _GRAPHICS_VOLUME_H_