#pragma once

#include "WHStream.h"

namespace whirl {

class WHMemoryChunk
{
public:
    WHMemoryChunk() = default;
    WHMemoryChunk(const void* src_ptr_set, void* dst_ptr_set, size_t size_set, const WHStream* stream_set = nullptr): 
        src_ptr_(src_ptr_set), dst_ptr_(dst_ptr_set), size_(size_set), stream_(stream_set) {}

    ~WHMemoryChunk()
    {
        src_ptr_ = dst_ptr_ = nullptr;
        size_ = 0;
        stream_ = nullptr;
    }

    const void* src_ptr() const { return src_ptr_; }
    void*       dst_ptr() const { return dst_ptr_; }
    
    size_t chunk_size() const { return size_; }

private:
    const void* src_ptr_;
    void*       dst_ptr_;
    size_t      size_;
    
    const WHStream* stream_;
};

}//namespace whirl