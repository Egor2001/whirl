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

struct WHMaterial
{
    WHColor mat_color;
    float   mat_transparency;
    float   mat_refract_factor;
};

}//namespace whirl