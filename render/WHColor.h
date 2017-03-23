#pragma once
#include "Windows.h"

struct WHColor
{
public:
    using Byte_ = BYTE;

	WHColor() : r(0), g(0), b(0) {}

	WHColor(Byte_ r_set, Byte_ g_set, Byte_ b_set) : r(r_set), g(g_set), b(b_set) {}

	WHColor(COLORREF color_set) :
		r(GetRValue(color_set)),
		g(GetGValue(color_set)),
		b(GetBValue(color_set))
	{}

	operator COLORREF() const
	{
		return RGB(r, g, b);
	}

public:    
    static const WHColor BLACK, WHITE, RED, GREEN, BLUE;
    
public:    
	Byte_ r, g, b;
};

const WHColor WHColor::BLACK = WHColor(  0,   0,   0), 
              WHColor::WHITE = WHColor(255, 255, 255), 
              WHColor::RED   = WHColor(255,   0,   0), 
              WHColor::GREEN = WHColor(  0, 255,   0), 
              WHColor::BLUE  = WHColor(  0,   0, 255);

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
