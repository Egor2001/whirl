#pragma once

#define _USE_MATH_DEFINES

#include <math.h>

#include "math/WHVector.h"
#include "math/WHMatrix.h"
#include "math/WHQuaternion.h"

namespace whirl {

class WHCamera
{
public:
    WHCamera(): position_{0.0f, 0.0f, 0.0f}, dir_eye_{0.0f, 0.0f, 1.0f}, dir_up_{0.0f, 1.0f, 0.0f}, angle_{M_PI/2.0f} {};
    WHCamera(): position_
    
    virtual ~WHCamera()
    {
        position_ = {};
        dir_eye_ = dir_up_ = {};
        angle_ = 0.0f;
    }
    
    void rotate(const WHQuaternion& rotation_quaternion)
    {
        dir_eye_ *= rotation_quaternion;
        dir_up_  *= rotation_quaternion;
    }

    void translate(const WHVector3& translation_vector) { position_ += translation_vector; }

    void set_angle(float angle_set) { angle_ = angle_set; }

    WHMatrix3x3 get_matrix() const { return WHMatrix3x3(dir_eye_ * dir_up_, dir_up_, dir_eye_); }

private:
    WHVector3 position_;
    WHNormal3 dir_eye_, dir_up_;
    float     angle_;
};

}//namespace whirl