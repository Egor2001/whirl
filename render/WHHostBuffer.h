#pragma once

#include "Windows.h"
#include <memory>

#include <algorithm>

#include "WHColor.h"
#include "WHMemoryManager.h"

#include "cuda.h"
#include "cuda_runtime.h"

namespace whirl {

template<WHMemoryLocation MemLocation_> class WHBuffer;

template<>
class WHBuffer<WHMemoryLocation::CPU>
{
public:
    using Byte_  = uint8_t;
    using Size_  = struct { size_t cx, cy; };
    using Color_ = WHColor;
    
    friend class WHBuffer<WHMemoryLocation::GPU>;

    static std::shared_ptr<WHBaseMemoryManager> create_mem_manager(const Size_& alloc_size);
    
    friend void swap(WHBuffer&, WHBuffer&);

    WHBuffer(): mem_manager_(), byte_size_({}), color_buf_(nullptr) {};
    explicit WHBuffer(const Size_& pixel_size_set);

    template<WHMemoryLocation OtherMemLocation> WHBuffer                           (const WHBuffer<OtherMemLocation>&);
    template<WHMemoryLocation OtherMemLocation> WHBuffer<MemLocation_>& operator = (const WHBuffer<OtherMemLocation>&);
    
    WHBuffer             (WHBuffer&&);
    WHBuffer& operator = (WHBuffer&&);
    
    virtual ~WHBuffer();
    
    bool   set_pixel(size_t x, size_t y, Color_ color_set);
    Color_ get_pixel(size_t x, size_t y) const;

    const std::shared_ptr<WHBaseMemoryManager>& get_mem_manager() const { return mem_manager_; }
    Size_                                       get_pixel_size () const { return { byte_size_.cx/3, byte_size_.cy }; }//could return pixel_size_set + 3 if pixel_size_set%4 == 3
    const Byte_*                                get_byte_buffer() const { return color_buf_; }
    /*
    __host__ size_t set_bytes_to_dc  (HDC dc) const;
    __host__ size_t get_bytes_from_dc(HDC dc);
    */
private:
    std::shared_ptr<WHBaseMemoryManager> mem_manager_;

    Size_  byte_size_;
    Byte_* color_buf_;
};

std::shared_ptr<WHBaseMemoryManager> WHBuffer<WHMemoryLocation::CPU>::create_mem_manager(const Size_& alloc_size)
{
    return WHHostMemoryManager<WHAllocType::HOST>::instance();
}

WHBuffer<WHMemoryLocation::CPU>::WHBuffer(const Size_& pixel_size_set): 
    mem_manager_(create_mem_manager(pixel_size_set)),
    byte_size_  ({ pixel_size_set.cx*3 + pixel_size_set.cx%4, pixel_size_set.cy }),
    color_buf_  (nullptr)
{
    size_t bytes_count = byte_size_.cx*byte_size_.cy*sizeof(Byte_);
    
    color_buf_ = static_cast<Byte_*>(mem_manager_->allocate(bytes_count));
    mem_manager_->memory_set(color_buf_, 0xFF, bytes_count);
}

template<WHMemoryLocation OtherMemLocation> 
WHBuffer<WHMemoryLocation::CPU>::WHBuffer(const WHBuffer<OtherMemLocation>& copy_from):
    mem_manager_(create_mem_manager(copy_from.byte_size_)),
    byte_size_  (copy_from.byte_size_),
    color_buf_  (nullptr)
{
    size_t bytes_count = byte_size_.cx*byte_size_.cy*sizeof(Byte_);

    color_buf_ = static_cast<Byte_*>(mem_manager_->allocate(bytes_count));
    
    if (copy_from.color_buf_) 
        mem_manager_->memory_copy(color_buf_, copy_from.color_buf_, copy_from.get_mem_manager()->alloc_type(), bytes_count);
    else
        mem_manager_->memory_set(color_buf_, 0xFF, bytes_count);
}

template<WHMemoryLocation OtherMemLocation> 
WHBuffer<WHMemoryLocation::CPU>& WHBuffer<MemLocation_>::operator = (const WHBuffer<OtherMemLocation>& copy_from)
{
    size_t bytes_count = byte_size_.cx*byte_size_.cy*sizeof(WHBuffer<OtherMemLocation>::Byte_);
    
    if (color_buf_) mem_manager_->deallocate(color_buf_);
    
    mem_manager_ = create_mem_manager(copy_from.byte_size_);
    byte_size_   = copy_from.byte_size_;
    color_buf_   = copy_from.color_buf_ ? static_cast<Byte_*>(mem_manager_->allocate(bytes_count)) : nullptr;
    
    if (copy_from.color_buf_) mem_manager_->memory_copy(color_buf_, copy_from.color_buf_, copy_from.get_mem_manager()->alloc_type(), bytes_count);

    return *this;
}

WHBuffer<WHMemoryLocation::CPU>::WHBuffer(WHBuffer<WHMemoryLocation::CPU>&& move_from):
    mem_manager_(std::move(move_from.mem_manager_)),
    byte_size_  (move_from.byte_size_),
    color_buf_  (move_from.color_buf_)
{
    move_from.byte_size_ = {};
    move_from.color_buf_ = nullptr;
}

WHBuffer<WHMemoryLocation::CPU>& WHBuffer<WHMemoryLocation::CPU>::operator = (WHBuffer<WHMemoryLocation::CPU>&& move_from)
{
    if (color_buf_) mem_manager_.deallocate(color_buf_);
    
    mem_manager_ = std::move(move_from.mem_manager_);
    byte_size_   = move_from.byte_size_; move_from.byte_size_ = {};
    color_buf_   = move_from.color_buf_; move_from.color_buf_ = nullptr;
    
    return *this;
}

void swap(WHBuffer<WHMemoryLocation::CPU>& lhs, WHBuffer<WHMemoryLocation::CPU>& rhs)
{
    std::swap(lhs.mem_manager_, rhs.mem_manager_);
    std::swap(lhs.byte_size_,   rhs.byte_size_);
    std::swap(lhs.color_buf_,   rhs.color_buf_);
}

WHBuffer<WHMemoryLocation::CPU>::~WHBuffer()
{
    if (color_buf_)
    {
        mem_manager_->deallocate(color_buf_);
        color_buf_ = nullptr;
    }

    byte_size_ = {};
}

bool WHBuffer<WHMemoryLocation::CPU>::set_pixel(size_t x, size_t y, Color_ color_set)
{
    if (x >= byte_size_.cx || y>= byte_size_.cy) return false;

    color_buf_[y*byte_size_.cx + x*3 + 0] = color_set.b;
    color_buf_[y*byte_size_.cx + x*3 + 1] = color_set.g;
    color_buf_[y*byte_size_.cx + x*3 + 2] = color_set.r;    
    
    return true;
}

template<WHMemoryLocation WHMemoryLocation::CPU>
typename WHBuffer<WHMemoryLocation::CPU>::Color_ WHBuffer<MemLocation_>::get_pixel(size_t x, size_t y) const
{
    Size_ pixel_size = get_pixel_size();

    if (x >= pixel_size.cx || y >= pixel_size.cy) return Color_();

    return Color_(color_buf_[y*byte_size_.cx + x*3 + 2],
                  color_buf_[y*byte_size_.cx + x*3 + 1],
                  color_buf_[y*byte_size_.cx + x*3 + 0]);
}
/*
template<WHMemoryLocation MemLocation_>
__host__ size_t WHBuffer<MemLocation_>::set_bytes_to_dc(HDC dc) const
{
    return SetDIBits(dc, static_cast<HBITMAP>(GetCurrentObject(dc, OBJ_BITMAP)), 
                     0, bmp_info_.bmiHeader.biHeight, color_buf_, &bmp_info_, DIB_RGB_COLORS);
}

template<WHMemoryLocation MemLocation_>
__host__ size_t WHBuffer<MemLocation_>::get_bytes_from_dc(HDC dc)
{
    return GetDIBits(dc, static_cast<HBITMAP>(GetCurrentObject(dc, OBJ_BITMAP)), 
                     0, bmp_info_.bmiHeader.biHeight, color_buf_, &bmp_info_, DIB_RGB_COLORS);
}
*/

}//namespace whirl