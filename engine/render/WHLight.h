#pragma once

#include "WHColor.h"
#include "math/WHVector.h"

namespace whirl {

class WHLight
{
public:
    __host__ __device__ WHLight() = default;
    __host__ __device__ WHLight(const WHVector3& position_set, const WHColor& color_set):
        position_(position_set), color_(color_set) {}

    __host__ __device__ ~WHLight()
    {
        position_ = {};
        color_    = {};
    }
    
    __device__ WHColor get_point_color(const WHMaterial& material, const WHNormal3& normal, const WHRay& ray)//Phong model
    {
        WHNormal3 inv_light_dir = normalized(position_ - ray.position);

        float light_dot = fabs(dot(inv_light_dir, normal));
        float eye_dot   = dot(-ray.direction, reflected(inv_light_dir, normal));

        WHColor result = (color_ * material.mat_color) * light_dot;
        
        result += color_ * pow(std::max(eye_dot, 0.0f), material.mat_shininess);

        normalize(result);

        return result;
    }

private:
    WHVector3 position_;
    WHColor   color_;
};

}//namespace whirl