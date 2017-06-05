#pragma once

#include "Windows.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "math_functions.h"

namespace whirl {

typedef float4 WHColor;

const WHColor WHColor::BLACK = WHColor(0.0f, 0.0f, 0.0f), 
              WHColor::WHITE = WHColor(1.0f, 1.0f, 1.0f),
              WHColor::RED   = WHColor(1.0f, 0.0f, 0.0f), 
              WHColor::GREEN = WHColor(0.0f, 1.0f, 0.0f), 
              WHColor::BLUE  = WHColor(0.0f, 0.0f, 1.0f);

WHColor  whMakeColorFloat(float   x, float   y, float   z, float alpha = 1.0f);
WHColor  whMakeColorByte (uint8_t x, uint8_t y, uint8_t z, float alpha = 1.0f);
COLORREF whToColorref    (const WHColor& color);

WHColor whMakeColorFloat(float x, float y, float z, float alpha = 1.0f)
{
    return make_float4(x, y, z, alpha);
}

WHColor  whMakeColorByte (uint8_t x, uint8_t y, uint8_t z, float alpha = 1.0f)
{
    float factor = 1.0f/255.0f;

    return make_float4(float(x)*factor, float(y)*factor, float(z)*factor, alpha);
}

COLORREF whToColorref(const WHColor& color)
{
    return RGB(255.0f*color.x, 255.0f*color.y, 255.0f*color.z);
}


inline WHColor operator *  (const WHColor&  color,	        float factor);
inline WHColor operator *  (         float factor, const WHColor&  color);
inline WHColor operator *  (const WHColor&     c1, const WHColor&     c2);
inline    void operator *= (	  WHColor&  color,          float factor);
inline    void operator *= (const WHColor&     c1, const WHColor&     c2);

inline WHColor ellColorMix(const WHColor& c1, float factor, const WHColor& c2);

WHColor operator * (const WHColor& color, float factor)
{
	if (factor > 1.0f) factor = 1.0f;

	return WHColor(color.r*factor, color.g*factor, color.b*factor);
}

WHColor operator * (float factor, const WHColor& color)
{
	if (factor > 1.0f) factor = 1.0f;

	return WHColor(color.r*factor, color.g*factor, color.b*factor);
}

WHColor operator * (const WHColor& c1, const WHColor& c2)
{
	return WHColor((c1.r * c2.r) / 255, (c1.g * c2.g) / 255, (c1.b * c2.b) / 255);
}

void operator *= (WHColor& color, float factor)
{
	if (factor > 1.0f) factor = 1.0f;

	color.r *= factor;
	color.g *= factor;
	color.b *= factor;
}

void operator *= (WHColor& c1, const WHColor& c2)
{
	c1.r = (c1.r * c2.r) / 255;
	c1.g = (c1.g * c2.g) / 255;
	c1.b = (c1.b * c2.b) / 255;
}

WHColor ellColorMix(const WHColor& c1, float factor, const WHColor& c2)
{
	if (factor > 1.0f) factor = 1.0f;

	return WHColor((c1.r - c2.r)*factor + c2.r, (c1.g - c2.g)*factor + c2.g, (c1.b - c2.b)*factor + c2.b);
}

}//namespace whirl