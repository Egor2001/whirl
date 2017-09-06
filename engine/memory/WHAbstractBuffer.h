#pragma once

#include "../logger/WHLogger.h"
#include "WHMemoryChunk.h"

namespace whirl {

class WHBuffer
{
public:
    typedef uint8_t                     Byte_;
    typedef struct { uint32_t cx, cy; } Size_;

    WHBuffer() = default;
    //TODO: memset
    WHBuffer(const std::shared_ptr<WHAbstractMemoryManager>& mem_manager_set, 
             const std::shared_ptr<WHChunkBuffer>&           chunk_buffer_set, unsigned int alloc_flags): 
        mem_manager_(mem_manager_set), chunk_buffer_(chunk_buffer_set), alloc_chunks_vec_(std::vector<Byte_*>(chunk_buffer_->get_chunks_num()))
    {
        auto alloc_size = chunk_buffer_->get_alloc_size();

        auto allocate_func = [&, this, alloc_flags](size_t chunk_index, size_t height_shift, size_t height_size, cudaStream_t stream)
        {
            alloc_chunks_vec_[chunk_index] = static_cast<Byte_*>(mem_manager_->allocate(alloc_size.cx * height_size * sizeof(Byte_), alloc_flags));
        };

        chunk_buffer_->iterate(allocate_func);

        WHIRL_TRACE("creating buffer [alloc type: {0:d}, alloc size: [{1:d}, {2:d}]]", mem_manager_->alloc_type(), alloc_size.cx, alloc_size.cy);
    }

    virtual ~WHBuffer()
    {
        if (!alloc_chunks_vec_.empty())
        {
            auto deallocate_func = [this](size_t chunk_index, size_t height_shift, size_t height_size, cudaStream_t stream)
            {
                mem_manager_->deallocate(static_cast<void*>(alloc_chunks_vec_[chunk_index]));
            };

            chunk_buffer_->iterate(deallocate_func);
        }
                
        WHIRL_TRACE("releasing buffer [alloc type: {0:d}]", mem_manager_->alloc_type());
    }

    Byte_* get_allocated_chunk(size_t chunk_index)
    { 
        WHIRL_CHECK_LT(chunk_index, chunk_buffer_->get_chunks_num());

        return alloc_chunks_vec_[chunk_index]; 
    }
    
    const Byte_* get_allocated_chunk(size_t chunk_index) const noexcept
    { 
        WHIRL_CHECK_LT(chunk_index, chunk_buffer_->get_chunks_num());

        return alloc_chunks_vec_[chunk_index]; 
    }
    
    std::shared_ptr<WHChunkBuffer> get_chunk_buffer() const noexcept { return chunk_buffer_; }
        
    bool set_pixel(uint32_t x, uint32_t y, const WHColor& color_set)
    {
        WHIRL_CHECK_NOT_EQ(alloc_chunks_vec_.empty(), true);
        
        size_t chunk_index = -1; 
        size_t offset = chunk_buffer_->get_relative_byte_offset(x, y, &chunk_index);
        
        WHIRL_CHECK_NOT_EQ(chunk_index, static_cast<size_t>(-1));

        alloc_chunks_vec_[chunk_index][offset+0] = color_set.x*255;
        alloc_chunks_vec_[chunk_index][offset+1] = color_set.y*255;
        alloc_chunks_vec_[chunk_index][offset+2] = color_set.z*255;
    }

    WHColor get_pixel(uint32_t x, uint32_t y)
    {
        WHIRL_CHECK_NOT_EQ(alloc_chunks_vec_.empty(), true);
        
        size_t chunk_index = -1; 
        size_t offset = chunk_buffer_->get_relative_byte_offset(x, y, &chunk_index);
        
        WHIRL_CHECK_NOT_EQ(chunk_index, static_cast<size_t>(-1));

        return whMakeColorByte(alloc_chunks_vec_[chunk_index][offset+0], 
                               alloc_chunks_vec_[chunk_index][offset+1], 
                               alloc_chunks_vec_[chunk_index][offset+2], 1.0f);
    }

protected:
    std::shared_ptr<WHAbstractMemoryManager> mem_manager_;
    std::shared_ptr<WHChunkBuffer>           chunk_buffer_;
    
    std::vector<Byte_*> alloc_chunks_vec_;
};

}//namespace whirl