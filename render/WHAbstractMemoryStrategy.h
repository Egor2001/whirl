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
    WHAbstractMemoryStrategy(const std::shared_ptr<WHAbstractMemoryManager>& memory_manager, 
                             const std::shared_ptr<WHChunkBuffer>&           chunk_buffer_set, unsigned int alloc_flags):
        chunk_buffer_(chunk_buffer_set), render_buffer_{memory_manager, chunk_buffer_, alloc_flags} {}

    virtual ~WHAbstractMemoryStrategy() = default;

    virtual void process_buffer(const WHScene* /* pointer to scene structure and other parameters */) = 0;
    
          WHBuffer& get_buffer()       { return render_buffer_; }//TODO: add synchronization via events or callbacks
    const WHBuffer& get_buffer() const { return render_buffer_; }//TODO: add synchronization via events or callbacks
        
protected:
    std::shared_ptr<WHChunkBuffer> chunk_buffer_;

    WHBuffer render_buffer_;
};

class WHSimpleMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHSimpleMemoryStrategy(const WHBuffer::Size_& buf_size): 
        host_mem_manager_  (WHMemoryManager<WHAllocType::HOST>  ::instance()),
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_{}, WHAbstractMemoryStrategy(std::dynamic_pointer_cast<WHAbstractMemoryManager>(host_mem_manager_), 
                                                   WHStreamManager::instance()->get_chunks({ buf_size.cx, buf_size.cy }), 0u)
    {
        device_buffer_ = WHBuffer(std::dynamic_pointer_cast<WHAbstractMemoryManager>(host_mem_manager_), chunk_buffer_, 0u);
    }

    virtual ~WHSimpleMemoryStrategy() override = default;
    
    virtual void process_buffer(const WHScene* scene_ptr) override
    {
        size_t alloc_width = chunk_buffer_->get_alloc_size().cx;

        //TODO: explicitly inline lambdas like this in WHChunkBuffer::iterate method or search for compiler optimization for it
        auto memory_operation = [this, scene_ptr, alloc_width](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            host_mem_manager_->copy_to_device(device_buffer_.get_allocated_chunk(index), 
                                              render_buffer_.get_allocated_chunk(index), height_size*alloc_width);
            
            //wh_launch_kernel(device_buffer_, device_buffer_.get_allocated_chunk(index), height_shift, height_size, scene_ptr);
        
            host_mem_manager_->copy_from_device(render_buffer_.get_allocated_chunk(index), 
                                                device_buffer_.get_allocated_chunk(index), height_size*alloc_width);
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::HOST>>   host_mem_manager_;
    std::shared_ptr<WHMemoryManager<WHAllocType::DEVICE>> device_mem_manager_;
    
    WHBuffer device_buffer_;
};

class WHPinnedMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHPinnedMemoryStrategy(const WHBuffer::Size_& buf_size, unsigned int alloc_flags = 0u):
        pinned_mem_manager_(WHMemoryManager<WHAllocType::PINNED>::instance()), 
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_{}, WHAbstractMemoryStrategy(std::dynamic_pointer_cast<WHAbstractMemoryManager>(pinned_mem_manager_), 
                                                   WHStreamManager::instance()->get_chunks({ buf_size.cx, buf_size.cy }), alloc_flags)
    {
        device_buffer_ = WHBuffer(std::dynamic_pointer_cast<WHAbstractMemoryManager>(device_mem_manager_), chunk_buffer_, 0u);
    }
    
    virtual ~WHPinnedMemoryStrategy() override = default;

    //TODO: via settings determine order of processing (if device could overlap kernel executing and memory copying, use this way)
    virtual void process_buffer(const WHScene* scene_ptr) override
    {
        size_t alloc_width = chunk_buffer_->get_alloc_size().cx;

        auto memory_operation = [this, scene_ptr, alloc_width](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            pinned_mem_manager_->async_copy_to_device(device_buffer_.get_allocated_chunk(index), 
                                                      render_buffer_.get_allocated_chunk(index), height_size*alloc_width, stream_handle);

            //wh_launch_kernel(device_buffer_, device_buffer_.get_allocated_chunk(index), height_shift, height_size, scene_ptr, stream_handle);

            pinned_mem_manager_->async_copy_from_device(render_buffer_.get_allocated_chunk(index), 
                                                        device_buffer_.get_allocated_chunk(index), height_size*alloc_width, stream_handle);
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::PINNED>> pinned_mem_manager_;
    std::shared_ptr<WHMemoryManager<WHAllocType::DEVICE>> device_mem_manager_;
    
    WHBuffer device_buffer_;
};
//TODO: find info about cases where cudaDeviceSynchronize must be used
class WHManagedMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHManagedMemoryStrategy(const WHBuffer::Size_& buf_size, int device_set, unsigned int alloc_flags = 0u, unsigned int attach_flags = cudaMemAttachSingle):
        managed_mem_manager_(WHMemoryManager<WHAllocType::MANAGED>::instance()), device_(device_set),
        WHAbstractMemoryStrategy(std::dynamic_pointer_cast<WHAbstractMemoryManager>(managed_mem_manager_),
                                 WHStreamManager::instance()->get_chunks({ buf_size.cx, buf_size.cy }), alloc_flags)
    {
        auto memory_operation = [this, attach_flags](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            managed_mem_manager_->attach_async(stream_handle, render_buffer_.get_allocated_chunk(index), attach_flags);
        };
        
        chunk_buffer_->iterate(memory_operation);

        WHIRL_CUDA_CALL(cudaDeviceSynchronize());
    }
    
    virtual ~WHManagedMemoryStrategy() override
    {
        auto memory_operation = [this](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            managed_mem_manager_->attach_async(stream_handle, render_buffer_.get_allocated_chunk(index), cudaMemAttachHost);
        };
        
        chunk_buffer_->iterate(memory_operation);

        WHIRL_CUDA_CALL(cudaDeviceSynchronize());

        device_ = 0;
    }

    //TODO: via settings determine order of processing (if device could overlap kernel executing and memory copying, use this way)
    virtual void process_buffer(const WHScene* scene_ptr) override
    {
        size_t alloc_width = chunk_buffer_->get_alloc_size().cx;

        auto memory_operation = [this, alloc_width](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            managed_mem_manager_->prefetch_to_device(render_buffer_.get_allocated_chunk(index), height_size*alloc_width, device_, stream_handle);
            //wh_launch_kernel(device_buffer_, render_buffer_.get_allocated_chunk(index), height_shift, height_size, scene_ptr, stream_handle);
            managed_mem_manager_->prefetch_to_host(render_buffer_.get_allocated_chunk(index), height_size*alloc_width, stream_handle);
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization

        WHIRL_CUDA_CALL(cudaDeviceSynchronize());
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::MANAGED>> managed_mem_manager_;
    
    int device_;
};

}//namespace whirl