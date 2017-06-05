#pragma once

#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"
#include "WHColor.h"

namespace whirl {

class WHAbstractBuffer abstract
{
public:
    typedef uint8_t                     Byte_;
    typedef struct { uint32_t cx, cy; } Size_;

    __host__ __device__ WHAbstractBuffer() = default;
    __host__ __device__ WHAbstractBuffer(const Size_& byte_buffer_size_set, Byte_* byte_buffer_ptr_set): 
                            byte_buffer_size_(byte_buffer_size_set), byte_buffer_ptr_(byte_buffer_ptr_set) {}

    __host__ __device__ virtual ~WHAbstractBuffer() = default;

    __host__ __device__       Byte_* get_byte_buffer  ()       { return byte_buffer_ptr_; }
    __host__ __device__ const Byte_* get_byte_buffer  () const { return byte_buffer_ptr_; }
    __host__ __device__ const Size_  get_size_in_bytes() const { return byte_buffer_size_; }

    __host__ __device__ bool set_pixel(uint32_t x, uint32_t y, const WHColor& color_set)
    {
        WHIRL_CHECK_NOT_EQ(byte_buffer_ptr_, nullptr);
        WHIRL_CHECK_LT    (3*x, byte_buffer_size_.cx);
        WHIRL_CHECK_LT    (y,   byte_buffer_size_.cy);

        uint32_t offset = y*byte_buffer_size_.cy + 3*x;
        
        byte_buffer_ptr_[offset+0] = color_set.x*255;
        byte_buffer_ptr_[offset+1] = color_set.y*255;
        byte_buffer_ptr_[offset+2] = color_set.z*255;
    }

    __host__ __device__ WHColor get_pixel(uint32_t x, uint32_t y)
    {
        WHIRL_CHECK_NOT_EQ(byte_buffer_ptr_, nullptr);
        WHIRL_CHECK_LT    (3*x, byte_buffer_size_.cx);
        WHIRL_CHECK_LT    (y,   byte_buffer_size_.cy);

        uint32_t offset = y*byte_buffer_size_.cy + 3*x;

        return whMakeColorByte(byte_buffer_ptr_[offset+0], 
                               byte_buffer_ptr_[offset+1], 
                               byte_buffer_ptr_[offset+2], 1.0f);
    }

protected:
    Size_  byte_buffer_size_;
    Byte_* byte_buffer_ptr_;
};

}//namespace whirl