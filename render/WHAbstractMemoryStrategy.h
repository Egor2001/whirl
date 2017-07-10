#pragma once

#include <memory>
#include <vector>

#include "WHAbstractBuffer.h"
#include "WHStream.h"
#include "WHMemoryManager.h"
#include "WHStreamManager.h"

namespace whirl {

class WHScene;//temp realization

class WHAbstractMemoryStrategy abstract
{
public:
    WHAbstractMemoryStrategy() = default;
    WHAbstractMemoryStrategy(const WHBuffer& render_buffer_set): render_buffer_(render_buffer_set) {}

    virtual ~WHAbstractMemoryStrategy() = default;

    virtual void process_buffer(const WHScene* /* pointer to scene structure and other parameters */) = 0;
    
          WHBuffer& get_buffer()       { return render_buffer_; }//TODO: add synchronization via events or callbacks
    const WHBuffer& get_buffer() const { return render_buffer_; }//TODO: add synchronization via events or callbacks
        
protected:
    WHBuffer render_buffer_;
};

class WHSimpleMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHSimpleMemoryStrategy(const WHBuffer::Size_& buf_size): 
        host_mem_manager_  (WHMemoryManager<WHAllocType::HOST>  ::instance()),
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_     {std::dynamic_pointer_cast<WHAbstractMemoryManager>(device_mem_manager_), buf_size, 0u}, buf_length_(0u),
        WHAbstractMemoryStrategy(WHBuffer(std::dynamic_pointer_cast<WHAbstractMemoryManager>(host_mem_manager_), buf_size, 0u)) 
    {
        auto device_buf_size = device_buffer_.get_size_in_bytes();
        auto render_buf_size = render_buffer_.get_size_in_bytes();

        buf_length_ = device_buf_size.cx * device_buf_size.cy;
        WHIRL_CHECK_EQ(buf_length_, render_buf_size.cx * render_buf_size.cy);
    }

    virtual ~WHSimpleMemoryStrategy() override
    {
        buf_length_ = 0u;
    }

    virtual void process_buffer(const WHScene* scene_ptr) override
    {
        host_mem_manager_->copy_to_device(device_buffer_.get_byte_buffer(), render_buffer_.get_byte_buffer(), buf_length_);
        
        //wh_launch_kernel(device_buffer_, device_buffer_.get_byte_buffer(), buf_length_, scene_ptr);
        
        host_mem_manager_->copy_from_device(render_buffer_.get_byte_buffer(), device_buffer_.get_byte_buffer(), buf_length_);
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::HOST>>   host_mem_manager_;
    std::shared_ptr<WHMemoryManager<WHAllocType::DEVICE>> device_mem_manager_;
    
    WHBuffer device_buffer_;

    size_t buf_length_;
};

class WHPinnedMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHPinnedMemoryStrategy(const WHBuffer::Size_& buf_size, unsigned int alloc_flags = 0u):
        pinned_mem_manager_(WHMemoryManager<WHAllocType::PINNED>::instance()), 
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_     {std::dynamic_pointer_cast<WHAbstractMemoryManager>(device_mem_manager_), buf_size, 0u},
        memory_chunk_buf_  {},
        WHAbstractMemoryStrategy(WHBuffer(std::dynamic_pointer_cast<WHAbstractMemoryManager>(pinned_mem_manager_), buf_size, alloc_flags))
    {
        auto buf_pixel_size = render_buffer_.get_size_in_pixels();

        memory_chunk_buf_ = WHStreamManager::instance()->get_chunks(buf_pixel_size.cx * buf_pixel_size.cy);//TODO: fix for returning aligned mod 3
    }
    
    virtual ~WHPinnedMemoryStrategy() override = default;

    //TODO: via settings determine order of processing (if device could overlap kernel executing and memory copying, use this way)
    virtual void process_buffer(const WHScene* scene_ptr) override
    {
        WHBuffer::Byte_* device_buf_ptr = device_buffer_.get_byte_buffer();
        WHBuffer::Byte_* render_buf_ptr = render_buffer_.get_byte_buffer();

        auto memory_operation = [&](ptrdiff_t shift, size_t size, cudaStream_t stream_handle)
        {
            pinned_mem_manager_->async_copy_to_device(device_buf_ptr + shift, render_buf_ptr + shift, size, stream_handle);
            //wh_launch_kernel(device_buffer_, device_buf_ptr + shift, size, scene_ptr);
            pinned_mem_manager_->async_copy_from_device(render_buf_ptr + shift, device_buf_ptr + shift, size, stream_handle);
        };

        memory_chunk_buf_.iterate(memory_operation);//must cause lambda - optimization
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::PINNED>> pinned_mem_manager_;
    std::shared_ptr<WHMemoryManager<WHAllocType::DEVICE>> device_mem_manager_;
    
    WHBuffer device_buffer_;

    WHChunkBuffer memory_chunk_buf_;
};

}//namespace whirl