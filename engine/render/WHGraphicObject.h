#pragma once

#include "math\WHVector.h"
#include "math\WHMatrix.h"
#include "WHMaterial.h"

namespace whirl {

struct WHGraphicObject
{
    __device__ WHGraphicObject(): material{WH_WHITE, 1.0f, 1.0f}, position{}, reverse_transform{} {}
    
    __device__ float device_distance_function(const WHVector3& point) const//simple SDF
    {
        
    }

    __device__ WHNormal3 device_get_normal(const WHVector3& position) const//takes gradient
    {

    }

    WHMaterial  material;
    WHVector3   position;
    WHMatrix3x3 reverse_transform;
};

}//namespace whirl