#pragma once

#include <memory>
#include <algorithm>

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

#include "../render/WHColor.h"
#include "../render/WHMemoryManager.h"

#include "../math/WHVector.h"

#include "WHMaterial.h"

namespace whirl {

struct WHRay
{
public:
    __device__ WHRay(const WHVector3& position_set, const WHNormal3& direction_set):
        position(position_set), direction(normalized(direction_set)), cur_refract_factor(1.0f) {}

    __device__ void step(float dist) 
    { 
        position += direction*dist; 
    }

    __device__ void refract(const WHMaterial& material, const WHNormal3& normal) //TODO: to test this function! 
    {
        float dot_ray_normal = dot(direction, normal)*cur_refract_factor; // must be positive value <=> cos > 0

        direction = (normal*(sqrtf(dot_ray_normal              * dot_ray_normal +
                                   material.mat_refract_factor * material.mat_refract_factor - 
                                   cur_refract_factor          * cur_refract_factor) - dot_ray_normal) +
                     direction*cur_refract_factor) * (1.0f/material.mat_refract_factor); // after that module of ray vector must be 1.0f

        cur_refract_factor = material.mat_refract_factor;
    }

public:
    WHVector3 position;
    WHNormal3 direction;
    float     cur_refract_factor;
};

}//namespace whirl