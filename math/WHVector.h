#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

namespace whirl {

typedef float3 WHVector3;
typedef float3 WHNormal3;

__host__ __device__ WHVector3 whMakeVector3(float x_set, float y_set, float z_set)
{
    return make_float3(x_set, y_set, z_set);
}

__host__ __device__ WHNormal3 whMakeNormal3(float x_set, float y_set, float z_set)
{
    return normalized(whMakeVector3(x_set, y_set, z_set));
}

__host__ __device__ WHVector3 operator - (const WHVector3&);
__host__ __device__ float     len        (const WHVector3&);
__host__ __device__ float     len2       (const WHVector3&);
__host__ __device__ WHVector3 normalized (const WHVector3&);
__host__ __device__ float     dot        (const WHVector3&, const WHVector3&);
__host__ __device__ WHVector3 mul        (const WHVector3&, const WHVector3&);

__host__ __device__ WHVector3 operator + (const WHVector3&, const WHVector3&);
__host__ __device__ WHVector3 operator - (const WHVector3&, const WHVector3&);
__host__ __device__ WHVector3 operator * (const WHVector3&, float);

__host__ __device__ WHVector3& operator += (WHVector3&, const WHVector3&);
__host__ __device__ WHVector3& operator -= (WHVector3&, const WHVector3&);
__host__ __device__ WHVector3& operator *= (WHVector3&, float);
__host__ __device__ void       normalize   (WHVector3&);

__host__ __device__ WHVector3 operator - (const WHVector3& vec)
{
    return make_float3(-vec.x, -vec.y, -vec.z);
}

__host__ __device__ float len(const WHVector3& vec)
{
    return sqrtf(len2(vec));
}

__host__ __device__ float len2(const WHVector3& vec)
{
    return dot(vec, vec);
}

__host__ __device__ WHVector3 normalized(const WHVector3& vec)//TODO: check if len is rather small
{
    return vec*(1.0f/len(vec));
}

__host__ __device__ float dot(const WHVector3& lhs_vec, const WHVector3& rhs_vec)
{
    return lhs_vec.x*rhs_vec.x + lhs_vec.y*rhs_vec.y + lhs_vec.z*rhs_vec.z; 
}

__host__ __device__ WHVector3 mul(const WHVector3& lhs_vec, const WHVector3& rhs_vec)
{
    return make_float3( lhs_vec.y*rhs_vec.z - lhs_vec.z*rhs_vec.y,
                       -lhs_vec.x*rhs_vec.z + lhs_vec.z*rhs_vec.x,
                        lhs_vec.x*rhs_vec.y - lhs_vec.y*rhs_vec.x);
}

__host__ __device__ WHVector3 operator + (const WHVector3& lhs_vec, const WHVector3& rhs_vec)
{
    return make_float3(lhs_vec.x + rhs_vec.x, 
                       lhs_vec.y + rhs_vec.y, 
                       lhs_vec.z + rhs_vec.z);
}

__host__ __device__ WHVector3 operator - (const WHVector3& lhs_vec, const WHVector3& rhs_vec)
{
    return make_float3(lhs_vec.x - rhs_vec.x, 
                       lhs_vec.y - rhs_vec.y, 
                       lhs_vec.z - rhs_vec.z);
}

__host__ __device__ WHVector3 operator * (const WHVector3& vec, float scale)
{
    return make_float3(vec.x*scale, vec.y*scale, vec.z*scale);
}

__host__ __device__ WHVector3& operator += (WHVector3& vec, const WHVector3& add)
{
    vec.x += add.x; vec.y += add.y; vec.z += add.z;

    return vec;
}

__host__ __device__ WHVector3& operator -= (WHVector3& vec, const WHVector3& sub)
{
    vec.x -= sub.x; vec.y -= sub.y; vec.z -= sub.z;

    return vec;
}

__host__ __device__ WHVector3& operator *= (WHVector3& vec, float scale)
{
    vec.x *= scale; vec.y *= scale; vec.z *= scale;

    return vec;
}

__host__ __device__ void normalize(WHVector3& vec)
{
    vec *= (1.0f/len(vec));
}

}//namespace whirl