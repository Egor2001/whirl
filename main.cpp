//#include "render\WHWindow.h"
//#include "render\WHBuffer.h"

#define SPDLOG_TRACE_ON

#include "stdio.h"

#include "render/WHMemoryManager.h"

#include "render/WHBuffer.h"
#include "render/WHWindow.h"

using namespace whirl;

int main()
{
    WHWindow wnd({ 256, 256 });

    WHBuffer<WHMemoryLocation::CPU> wnd_buffer = wnd.create_buffer();
    WHBuffer<WHMemoryLocation::GPU> gpu_buffer = wnd_buffer;
    
    auto buf_size = wnd_buffer.get_pixel_size();
    
    for(size_t x = 0; x < buf_size.cx; x++)
    for(size_t y = 0; y < buf_size.cy; y++)
    {
        wnd_buffer.set_pixel(x, y, WHColor::BLUE);
    }
    
    wnd.flush(&wnd_buffer);
    
    return 0;
}