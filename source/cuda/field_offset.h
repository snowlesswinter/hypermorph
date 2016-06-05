#ifndef _FIELD_OFFSET_H_
#define _FIELD_OFFSET_H_

#include "cuda_runtime.h"

float3 GetOffsetVorticityField(int n)
{
    float3 result[] = {
        make_float3(0.0f, 0.5f, 0.5f),
        make_float3(0.5f, 0.0f, 0.5f),
        make_float3(0.5f, 0.5f, 0.0f)
    };
    if (n >= sizeof(result) / sizeof(result[0]))
        return make_float3(0.0f);

    return result[n];
}

float3 GetOffsetVelocityField(int n)
{
    float3 result[] = {
        make_float3(0.5f, 0.0f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f),
        make_float3(0.0f, 0.0f, 0.5f)
    };
    if (n >= sizeof(result) / sizeof(result[0]))
        return make_float3(0.0f);

    return result[n];
}

float3 CombineOffsets(float3 backwards, float3 forwards)
{
    return forwards - backwards;
}

#endif // _FIELD_OFFSET_H_