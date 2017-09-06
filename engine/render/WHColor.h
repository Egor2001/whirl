#pragma once

#include <cstdint>

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

#include "../logger/WHLogger.h"

namespace whirl {

typedef float4 WHColor;//color realization optimized for direct ray marching with color overlaying (w is not similar to alpha)

const WHColor WH_BLACK {0.0f, 0.0f, 0.0f, 1.0f}, 
              WH_WHITE {1.0f, 1.0f, 1.0f, 1.0f},
              WH_RED   {1.0f, 0.0f, 0.0f, 1.0f}, 
              WH_GREEN {0.0f, 1.0f, 0.0f, 1.0f}, 
              WH_BLUE  {0.0f, 0.0f, 1.0f, 1.0f};

__host__ __device__ WHColor whMakeColorFloat(float   x, float   y, float   z, float alpha = 1.0f);
__host__ __device__ WHColor whMakeColorByte (uint8_t x, uint8_t y, uint8_t z, float alpha = 1.0f);

__host__ __device__ WHColor whMakeColorFloat(float x, float y, float z, float alpha = 1.0f)
{
    return make_float4(x, y, z, alpha);
}

__host__ __device__ WHColor whMakeColorByte(uint8_t x, uint8_t y, uint8_t z, float alpha = 1.0f)
{
    float factor = 1.0f/255.0f;

    return make_float4(float(x)*factor, float(y)*factor, float(z)*factor, alpha);
}

__host__ __device__ inline uchar3 whGetByteColor(const WHColor&);

__host__ __device__ inline void     truncate    (      WHColor&);
__host__ __device__ inline void     normalize   (      WHColor&);
__host__ __device__ inline WHColor  operator *  (const WHColor&,	      float);
__host__ __device__ inline WHColor  operator *  (         float, const WHColor&);
__host__ __device__ inline WHColor  operator *  (const WHColor&, const WHColor&);
__host__ __device__ inline WHColor  operator +  (const WHColor&, const WHColor&);
__host__ __device__ inline WHColor& operator *= (	   WHColor&,          float);
__host__ __device__ inline WHColor& operator *= (      WHColor&, const WHColor&);
__host__ __device__ inline WHColor& operator += (      WHColor&, const WHColor&);

__host__ __device__ inline uchar3 whGetByteColor(const WHColor& color)
{
    if (color.w < FLT_EPSILON) return {0u, 0u, 0u};

    float factor = 1.0f/color.w;

    return {uint8_t(color.x*factor*255.0f), uint8_t(color.y*factor*255.0f), uint8_t(color.z*factor*255.0f)};
}

__host__ __device__ inline void truncate(WHColor& color)
{
    normalize(color);

    color.x = std::max(color.x, 1.0f);
    color.y = std::max(color.y, 1.0f);
    color.z = std::max(color.z, 1.0f);
}

__host__ __device__ void normalize(WHColor& color)
{
    if (color.w < FLT_EPSILON) color = WH_BLACK;

    color *= (1.0f / color.w);
}

__host__ __device__ WHColor operator * (const WHColor& color, float factor)
{
	WHIRL_CHECK_LTEQ(fabs(factor), 1.0f);

    return {color.x*factor, color.y*factor, color.z*factor, color.w*factor};
}

__host__ __device__ WHColor operator * (float factor, const WHColor& color)
{
	WHIRL_CHECK_LTEQ(fabs(factor), 1.0f);

    return {color.x*factor, color.y*factor, color.z*factor, color.w*factor};
}

__host__ __device__ WHColor operator * (const WHColor& lhs, const WHColor& rhs)
{
    return {lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z, lhs.w*rhs.w};
}

__host__ __device__ WHColor operator + (const WHColor& lhs, const WHColor& rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
}

__host__ __device__ WHColor& operator *= (WHColor& color, float factor)
{
	WHIRL_CHECK_LTEQ(fabs(factor), 1.0f);

    return (color = {color.x*factor, color.y*factor, color.z*factor, color.w*factor});
}

__host__ __device__ WHColor& operator *= (WHColor& lhs, const WHColor& rhs)
{
    return (lhs = {lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z, lhs.w*rhs.w});
}

__host__ __device__ WHColor& operator += (WHColor& lhs, const WHColor& rhs)
{
    return (lhs = {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w});
}

}//namespace whirl