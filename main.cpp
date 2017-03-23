//#include "render\WHWindow.h"
//#include "render\WHBuffer.h"

#define SPDLOG_TRACE_ON

#include <cstdio>

//#include "render\WHMemoryManager.h"

#include "render/WHBuffer.h"

int main()
{
    //auto cpu_buffer = WHBuffer<WHMemoryLocation::CPU>({ 256, 256});
    //WHBuffer<WHMemoryLocation::GPU> gpu_buffer = cpu_buffer;
    
    auto buffer = WHBuffer<WHMemoryLocation::GPU>({ 256, 256 });

    return 0;
}