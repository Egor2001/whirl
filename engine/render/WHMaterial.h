#pragma once

#include "../render/WHColor.h"

namespace whirl {

struct WHMaterial
{
    WHColor mat_color;
    float   mat_shininess;
    float   mat_transparency;
    float   mat_refract_factor;
};

}//namespace whirl