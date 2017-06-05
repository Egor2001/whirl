#pragma once

#include <memory>

#include <algorithm>

#include "../render/WHColor.h"
#include "../render/WHMemoryManager.h"

#include "../math/WHVector.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

namespace whirl {

class WHLight
{
public:
    __host__ __device__ WHLight() = default;
    __host__ __device__ WHLight(const WHVector3& position_set, const WHColor& color_set):
        position_(position_set), color_(color_set) {}

    __host__ __device__ ~WHLight() = default;
    
    __device__ WHColor get_point_color(const WHVector3& point, const WHNormal3& normal);

private:
    WHVector3 position_;
    WHColor   color_;
};

}//namespace whirl