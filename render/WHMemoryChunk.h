#pragma once

#include <ctype.h>

#include "WHStream.h"

namespace whirl {

class WHMemoryChunk
{
public:
    WHMemoryChunk() = default;
    WHMemoryChunk(ptrdiff_t shift_set, size_t size_set, std::shared_ptr<WHStream> stream_set = nullptr): 
        shift_(shift_set), size_(size_set), stream_(stream_set) {}

    ~WHMemoryChunk()
    {
        shift_ = size_ = 0;
        stream_ = nullptr;
    }

    ptrdiff_t get_shift() const noexcept { return shift_; }
    size_t    get_size () const noexcept { return size_; }

private:
    ptrdiff_t shift_;
    size_t    size_;
    
    std::shared_ptr<WHStream> stream_;
};

}//namespace whirl