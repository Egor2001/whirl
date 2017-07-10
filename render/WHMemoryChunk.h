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
    
    template<typename Func_t>
    void memory_operation(Func_t&& function) const noexcept
    {
        //guaranteed not to change arguments via const - qualifier of this (WHMemoryChunk::memory_operation) function 
        std::forward<Func_t>(function)(shift_, size_, stream_->get_stream_handle());
    }

private:
    ptrdiff_t shift_;
    size_t    size_;
    
    std::shared_ptr<WHStream> stream_;
};

class WHChunkBuffer final
{
public:
    WHChunkBuffer() = default;
    WHChunkBuffer(const std::vector<WHMemoryChunk>&  chunk_vec_set): chunk_vec_(chunk_vec_set) {}
    WHChunkBuffer(      std::vector<WHMemoryChunk>&& chunk_vec_set): chunk_vec_(chunk_vec_set) {}

    WHChunkBuffer             (WHChunkBuffer&& set_from): chunk_vec_(std::move(set_from.chunk_vec_)) {}
    WHChunkBuffer& operator = (WHChunkBuffer&& set_from) { chunk_vec_ = std::move(set_from.chunk_vec_); }

    ~WHChunkBuffer() = default;
    
    void add_chunk(const WHMemoryChunk& chunk) { chunk_vec_.push_back(chunk); }

    template<typename Func_t>
    void iterate(Func_t&& func) const noexcept
    {
        for (const auto& chunk : chunk_vec_)
            chunk.memory_operation(func);
    }

private:
    std::vector<WHMemoryChunk> chunk_vec_;
};

}//namespace whirl