#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "memory/WHAbstractBuffer.h"
#include "memory/WHStream.h"
#include "render/WHColor.h"
#include "render/math/WHVector.h"
#include "WHAbstractScene.h"
#include "render/WHRay.h"

namespace whirl {

struct WHNormalQuad
{
    WHNormalQuad(): left_top   {-1.0f, -1.0f, 1.0f}, right_top   {1.0f, -1.0f, 1.0f}, 
                    left_bottom{-1.0f,  1.0f, 1.0f}, right_bottom{1.0f,  1.0f, 1.0f} {}

    WHNormalQuad(const WHVector3& bounds): 
        left_top   (whMakeNormal3(-bounds.x/2, -bounds.y/2, bounds.z)), right_top   (whMakeNormal3(bounds.x/2, -bounds.y/2, bounds.z)), 
        left_bottom(whMakeNormal3(-bounds.x/2,  bounds.y/2, bounds.z)), right_bottom(whMakeNormal3(bounds.x/2,  bounds.y/2, bounds.z)) {}

    WHNormal3 left_top,    right_top, 
              left_bottom, right_bottom; 
};

template<class SceneType>
__global__ void wh_kernel(uint8_t* allocated_chunk, const SceneType* scene_ptr, size_t chunk_byte_width, size_t chunk_height, WHNormalQuad volume_bound)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= chunk_byte_width/3 || y >= chunk_height) return;
    
    uchar3 byte_color = whGetByteColor(scene_ptr->device_trace_ray(wh_device_get_ray(volume_bound, x, y, chunk_byte_width/3 - x, chunk_height - y)));
        
    allocated_chunk[y*chunk_byte_width + 3*x + 0] = byte_color.x;
    allocated_chunk[y*chunk_byte_width + 3*x + 1] = byte_color.y;
    allocated_chunk[y*chunk_byte_width + 3*x + 2] = byte_color.z;
}

template<class SceneType>
cudaError_t wh_launch_kernel(const WHBuffer* buffer_ptr, uint8_t* allocated_chunk, const SceneType* scene_ptr, 
                             size_t height_shift, size_t height_size, cudaStream_t stream_handle = 0)//TODO: find out optimal parameters for kernel launcher
{
    WHIRL_CHECK_NOT_EQ(buffer_ptr,      nullptr);
    WHIRL_CHECK_NOT_EQ(allocated_chunk, nullptr);
    WHIRL_CHECK_NOT_EQ(scene_ptr,       nullptr);

    WHNormalQuad volume_bounds {whMakeNormal3(), whMakeNormal3(), 
                                whMakeNormal3(), whMakeNormal3()};//may be, it's better to alloc it on device side

    WHIRL_TRACE("launch kernel [shift: {0:d}, size: {1:d}, stream: {2:p}]", height_shift, height_size, stream_handle);

    size_t pixel_width = buffer_ptr->get_chunk_buffer()->get_pixel_size().cx;
    size_t alloc_width = buffer_ptr->get_chunk_buffer()->get_alloc_size().cx;

    dim3 block(16, 16), grid((pixel_width-1)/block.x + 1, (height_size-1)/block.y + 1);
    
    wh_kernel<SceneType> <<<grid, block, 0, stream_handle>>>
        (allocated_chunk, scene_ptr, alloc_width, height_size, volume_bounds);
}

}//namespace whirl