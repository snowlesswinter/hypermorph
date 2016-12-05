//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#ifndef _FLUID_BUFFER_OWNER_H_
#define _FLUID_BUFFER_OWNER_H_

#include "graphics_lib_enum.h"

class GraphicsLinearMemU16;
class GraphicsMemPiece;
class GraphicsVolume;
class GraphicsVolume3;
class FluidBufferOwner
{
public:
    virtual ~FluidBufferOwner() {}

    virtual GraphicsMemPiece* GetActiveParticleCountMemPiece() = 0;
    virtual GraphicsVolume* GetDensityVolume() = 0;
    virtual GraphicsVolume3* GetVelocityField() = 0;
    virtual GraphicsLinearMemU16* GetParticleDensityField() = 0;
    virtual GraphicsLinearMemU16* GetParticlePosXField() = 0;
    virtual GraphicsLinearMemU16* GetParticlePosYField() = 0;
    virtual GraphicsLinearMemU16* GetParticlePosZField() = 0;
    virtual GraphicsLinearMemU16* GetParticleTemperatureField() = 0;
    virtual GraphicsVolume* GetTemperatureVolume() = 0;

protected:
    FluidBufferOwner() {}
};

#endif // _FLUID_BUFFER_OWNER_H_