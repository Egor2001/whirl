#pragma once

#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"
#include "WHColor.h"

namespace whirl {

class WHBuffer
{
public:
    typedef uint8_t                     Byte_;
    typedef struct { uint32_t cx, cy; } Size_;

    WHBuffer() = default;
    
    WHBuffer(std::shared_ptr<WHAbstractMemoryManager> mem_manager_set, const Size_& pixel_size, unsigned int alloc_flags): 
        mem_manager_     (mem_manager_set), 
        byte_buffer_size_({ pixel_size.cx*3 + pixel_size.cx%4, pixel_size.cy }), 
        byte_buffer_ptr_ (static_cast<Byte_*>(mem_manager_->allocate(byte_buffer_size_.cx * byte_buffer_size_.cy, alloc_flags)))
    {
        WHIRL_TRACE("creating buffer [alloc type: {0:d}, size: [{1:d}, {2:d}]]", mem_manager_->alloc_type(), pixel_size.cx, pixel_size.cy);
    }

    virtual ~WHBuffer()
    {
        if (byte_buffer_ptr_)
            mem_manager_->deallocate(static_cast<void*>(byte_buffer_ptr_));
        
        byte_buffer_size_ = {};
        byte_buffer_ptr_  = nullptr;

        WHIRL_TRACE("releasing buffer [alloc type: {0:d}]", mem_manager_->alloc_type());
    }

          Byte_* get_byte_buffer   ()       { return byte_buffer_ptr_; }
    const Byte_* get_byte_buffer   () const { return byte_buffer_ptr_; }
    const Size_& get_size_in_bytes () const { return byte_buffer_size_; }
          Size_  get_size_in_pixels() const { return { byte_buffer_size_.cx/3, byte_buffer_size_.cy }; }

    bool set_pixel(uint32_t x, uint32_t y, const WHColor& color_set)
    {
        WHIRL_CHECK_NOT_EQ(byte_buffer_ptr_, nullptr);
        WHIRL_CHECK_LT    (3*x, byte_buffer_size_.cx);
        WHIRL_CHECK_LT    (y,   byte_buffer_size_.cy);

        uint32_t offset = y*byte_buffer_size_.cy + 3*x;
        
        byte_buffer_ptr_[offset+0] = color_set.x*255;
        byte_buffer_ptr_[offset+1] = color_set.y*255;
        byte_buffer_ptr_[offset+2] = color_set.z*255;
    }

    WHColor get_pixel(uint32_t x, uint32_t y)
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
    std::shared_ptr<WHAbstractMemoryManager> mem_manager_;

    Size_  byte_buffer_size_;
    Byte_* byte_buffer_ptr_;
};

}//namespace whirl