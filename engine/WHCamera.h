#pragma once

#include <memory>

#include <algorithm>

#include "../render/WHColor.h"
#include "../render/WHMemoryManager.h"

#include "../math/WHVector.h"
#include "../math/WHMatrix.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

namespace whirl {

class WHCamera
{
public:
    __host__ __device__ WHCamera();
    
    __host__ __device__ ~WHCamera() = default;
    
    __host__ WHMatrix3x3 get_matrix()
    {
        return WHMatrix3x3(mul(dir_eye_, dir_up_), dir_up_, dir_eye_);
    }

private:
    WHVector3 position_;
    WHNormal3 dir_eye_, dir_up_;
    float     angle;
};

}//namespace whirl