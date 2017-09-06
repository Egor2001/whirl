#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

#include "WHVector.h"

namespace whirl {

typedef float4 WHQuaternion;

__host__ __device__ float norm     (const WHQuaternion&);
__host__ __device__ float magnitude(const WHQuaternion&);

__host__ __device__ WHQuaternion conjugated(const WHQuaternion&);
__host__ __device__ WHQuaternion inverted  (const WHQuaternion&);

__host__ __device__ WHQuaternion operator * (const WHQuaternion&, const WHQuaternion&);
__host__ __device__ WHVector3    operator * (const WHVector3&,    const WHQuaternion&);
__host__ __device__ WHQuaternion operator * (const WHQuaternion&, float);

__host__ __device__ WHQuaternion& operator *= (WHQuaternion&, const WHQuaternion&);
__host__ __device__ WHVector3&    operator *= (WHVector3&,    const WHQuaternion&);
__host__ __device__ WHQuaternion& operator *= (WHQuaternion&, float);


}//namespace whirl