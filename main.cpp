//#include "render\WHWindow.h"
//#include "render\WHBuffer.h"

#define SPDLOG_TRACE_ON

#include <cstdio>

//#include "render\WHMemoryManager.h"

#include "render/WHBuffer.h"
#include "render/WHWindow.h"

int main()
{
    //auto cpu_buffer = WHBuffer<WHMemoryLocation::CPU>({ 256, 256});
    //WHBuffer<WHMemoryLocation::GPU> gpu_buffer = cpu_buffer;
    
    WHWindow wnd({ 256, 256 });

    WHBuffer<WHMemoryLocation::CPU> wnd_buf = wnd.create_buffer();

    auto buf_size = wnd_buf.get_pixel_size();
    
    for(size_t x = 0; x < buf_size.cx; x++)
    for(size_t y = 0; y < buf_size.cy; y++)
    {
        wnd_buf.set_pixel(x, y, WHColor::BLUE);
    }
    
    wnd.flush(&wnd_buf);

    return 0;
}