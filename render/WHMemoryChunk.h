#pragma once

#include <ctype.h>

#include "WHStream.h"

namespace whirl {

class WHMemoryChunk
{
public:
    WHMemoryChunk() = default;
    WHMemoryChunk(size_t index_set, size_t height_shift_set, size_t height_size_set, std::shared_ptr<WHStream> stream_set = nullptr): 
        index_(index_set), height_shift_(height_shift_set), height_size_(height_size_set), stream_(stream_set) {}

    ~WHMemoryChunk()
    {
        height_shift_ = height_size_ = 0;
        stream_ = nullptr;
    }

    size_t get_index() const noexcept { return index_; }
    size_t get_shift() const noexcept { return height_shift_; }
    size_t get_size () const noexcept { return height_size_; }
    
    template<typename Func_t>
    void memory_operation(Func_t&& function) const noexcept
    {
        //guaranteed not to change arguments via const - qualifier of this (WHMemoryChunk::memory_operation) function 
        std::forward<Func_t>(function)(index_, height_shift_, height_size_, stream_->get_stream_handle());
    }

private:
    size_t index_;

    size_t height_shift_;
    size_t height_size_;
    
    std::shared_ptr<WHStream> stream_;
};

class WHChunkBuffer
{
public:
    typedef struct { size_t cx, cy; } Size_;

    WHChunkBuffer() = default;
    WHChunkBuffer(const Size_& pixel_size_set, const std::vector<std::shared_ptr<WHStream>>& stream_vec): 
        pixel_size_(pixel_size_set), alloc_size_({ pixel_size_.cx*3 + pixel_size_.cx%4, pixel_size_.cy }), chunk_vec_{} 
    {
        size_t streams_num = stream_vec.size();

        chunk_vec_.resize(streams_num);
        
        float delta = float(alloc_size_.cy)/streams_num;

        for (size_t i = 0; i < streams_num; i++)
            chunk_vec_[i] = WHMemoryChunk(i, size_t(i*delta), size_t((i+1)*delta) - size_t(i*delta), stream_vec[i]);
    }
    
    WHChunkBuffer             (WHChunkBuffer&& set_from): chunk_vec_(std::move(set_from.chunk_vec_)) {}
    WHChunkBuffer& operator = (WHChunkBuffer&& set_from) { chunk_vec_ = std::move(set_from.chunk_vec_); }

    ~WHChunkBuffer() = default;
    
    const Size_& get_pixel_size() const noexcept { return pixel_size_; }
    const Size_& get_alloc_size() const noexcept { return alloc_size_; }
    size_t       get_chunks_num() const noexcept { return chunk_vec_.size(); }

    template<typename Func_t>
    void iterate(Func_t&& func) const noexcept
    {
        for (const auto& chunk : chunk_vec_)
            chunk.memory_operation(func);
    }

    size_t get_relative_byte_offset(size_t x, size_t y, size_t* chunk_index) const
    {
        WHIRL_CHECK_NOT_EQ(chunk_index, nullptr);
        WHIRL_CHECK_LT    (x, pixel_size_.cx);
        WHIRL_CHECK_LT    (y, pixel_size_.cy);

        size_t absolute_byte_offset = y*alloc_size_.cx + 3*x;

        auto compare_func = [this](const WHMemoryChunk& chunk, size_t absolute_byte_offset)
        { 
            return (chunk.get_shift() + chunk.get_size())*alloc_size_.cx < absolute_byte_offset; 
        };

        auto chunk_iter = std::lower_bound(chunk_vec_.cbegin(), chunk_vec_.cend(), absolute_byte_offset, compare_func);

        *chunk_index = chunk_iter->get_index();

        return absolute_byte_offset - chunk_iter->get_shift()*alloc_size_.cx;
    }

private:
    Size_ pixel_size_, alloc_size_;//Because it is not possible to determine pixel size via alloc size (but pixel size is used only to return it via getter)

    std::vector<WHMemoryChunk> chunk_vec_;
};

}//namespace whirl