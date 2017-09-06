#pragma once

#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

#include "WHVector.h"

namespace whirl {
    
struct WHMatrix3x3
{
    __host__ __device__ 
    WHMatrix3x3(): 
        elements {1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f} {}

    __host__ __device__ 
    WHMatrix3x3(const WHVector3& vec_e1, const WHVector3& vec_e2, const WHVector3& vec_e3):
        elements {vec_e1.x, vec_e1.y, vec_e1.z,
                    vec_e2.x, vec_e2.y, vec_e2.z,
                    vec_e3.x, vec_e3.y, vec_e3.z} {}

    __host__ __device__ 
    WHMatrix3x3(const WHMatrix3x3& src_matrix): elements() 
    {
        memcpy(elements, src_matrix.elements, 9); 
    }

    __host__ __device__ 
    WHMatrix3x3& operator = (const WHMatrix3x3& src_matrix)
    {
        memcpy(elements, src_matrix.elements, 9);

        return (*this);
    }

    __host__ __device__ ~WHMatrix3x3() = default;

    float elements[9];
};

__host__ __device__ WHMatrix3x3 operator *  (const WHMatrix3x3&, const WHMatrix3x3&);
__host__ __device__ WHVector3   operator *  (const WHVector3&,   const WHMatrix3x3&);

__host__ __device__ WHMatrix3x3& operator *= (WHMatrix3x3&, const WHMatrix3x3&);
__host__ __device__ WHVector3&   operator *= (WHVector3&,   const WHMatrix3x3&);
    
__host__ __device__ WHMatrix3x3 triangle(const WHMatrix3x3&);
    
__host__ __device__ float       det(const WHMatrix3x3&);
__host__ __device__ WHMatrix3x3 inv(const WHMatrix3x3&);

}//namespace whirl