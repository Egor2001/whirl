#pragma once

#include <memory>
#include <vector>

#include "memory/WHStreamManager.h"
#include "memory/WHAbstractBuffer.h"
#include "memory/WHEvent.h"
#include "WHAbstractScene.h"
#include "WHKernel.cuh"

namespace whirl {

class WHAbstractMemoryStrategy abstract//no need in polymorphism
{
public:
    WHAbstractMemoryStrategy() = default;
    WHAbstractMemoryStrategy(const std::shared_ptr<WHEvent>& synchronizer_set, const std::shared_ptr<WHAbstractMemoryManager>& memory_manager, 
                             const std::shared_ptr<WHChunkBuffer>& chunk_buffer_set, unsigned int alloc_flags):
        synchronizer_(synchronizer_set), chunk_buffer_(chunk_buffer_set), render_buffer_{memory_manager, chunk_buffer_, alloc_flags} {}

    virtual ~WHAbstractMemoryStrategy() = default;
    
    template<class SceneType>
    void process_buffer(const SceneType* scene_ptr)
    {
        add_callbacks();
    }
    
    void add_callbacks()
    {
        cudaStreamCallback_t callback = [](cudaStream_t stream, cudaError_t status, void* sync_event_ptr) -> void CUDART_CB 
        {
            WHIRL_TRACE("kernel finished [stream: {0:p}, status: {1:s}]", cudaGetErrorString(status));

            static_cast<WHEvent*>(sync_event_ptr)->record(stream);
        };
        
        auto memory_operation = [this,  callback](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            WHIRL_CUDA_CALL(cudaStreamAddCallback(stream_handle, callback, static_cast<void*>(synchronizer_.get()), 0u));
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization
    }

          std::shared_ptr<WHEvent>& get_synchronizer()       { return synchronizer_; }
    const std::shared_ptr<WHEvent>& get_synchronizer() const { return synchronizer_; }
    
          WHBuffer& get_buffer()       { return render_buffer_; }//TODO: add synchronization via events or callbacks
    const WHBuffer& get_buffer() const { return render_buffer_; }//TODO: add synchronization via events or callbacks
        
protected:
    std::shared_ptr<WHEvent> synchronizer_;

    std::shared_ptr<WHChunkBuffer> chunk_buffer_;
    WHBuffer                       render_buffer_;
};

class WHSimpleMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHSimpleMemoryStrategy(const std::shared_ptr<WHEvent>& synchronizer, const WHBuffer::Size_& buf_size): 
        host_mem_manager_  (WHMemoryManager<WHAllocType::HOST>  ::instance()),
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_{}, WHAbstractMemoryStrategy(synchronizer, std::dynamic_pointer_cast<WHAbstractMemoryManager>(host_mem_manager_), 
                                                   WHStreamManager::instance()->get_chunks({ buf_size.cx, buf_size.cy }), 0u)
    {
        device_buffer_ = WHBuffer(std::dynamic_pointer_cast<WHAbstractMemoryManager>(host_mem_manager_), chunk_buffer_, 0u);
    }

    virtual ~WHSimpleMemoryStrategy() override = default;
    
    template<class SceneType>
    void process_buffer(const SceneType* scene_ptr)
    {
        WHIRL_CHECK_NOT_EQ(scene_ptr, nullptr);
        
        size_t alloc_width = chunk_buffer_->get_alloc_size().cx;

        //TODO: explicitly inline lambdas like this in WHChunkBuffer::iterate method or search for compiler optimization for it
        auto memory_operation = [this, scene_ptr, alloc_width](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            host_mem_manager_->copy_to_device(device_buffer_.get_allocated_chunk(index), 
                                              render_buffer_.get_allocated_chunk(index), height_size*alloc_width);
            
            wh_launch_kernel<SceneType>(device_buffer_, device_buffer_.get_allocated_chunk(index), scene_ptr, height_shift, height_size);
        
            host_mem_manager_->copy_from_device(render_buffer_.get_allocated_chunk(index), 
                                                device_buffer_.get_allocated_chunk(index), height_size*alloc_width);
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization
        
        WHAbstractMemoryStrategy::process_buffer<SceneType>(scene_ptr);
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::HOST>>   host_mem_manager_;
    std::shared_ptr<WHMemoryManager<WHAllocType::DEVICE>> device_mem_manager_;
    
    WHBuffer device_buffer_;
};

class WHPinnedMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHPinnedMemoryStrategy(const std::shared_ptr<WHEvent>& synchronizer, const WHBuffer::Size_& buf_size, unsigned int alloc_flags = 0u):
        pinned_mem_manager_(WHMemoryManager<WHAllocType::PINNED>::instance()), 
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_{}, WHAbstractMemoryStrategy(synchronizer, std::dynamic_pointer_cast<WHAbstractMemoryManager>(pinned_mem_manager_), 
                                                   WHStreamManager::instance()->get_chunks({ buf_size.cx, buf_size.cy }), alloc_flags)
    {
        device_buffer_ = WHBuffer(std::dynamic_pointer_cast<WHAbstractMemoryManager>(device_mem_manager_), chunk_buffer_, 0u);
    }
    
    virtual ~WHPinnedMemoryStrategy() override = default;

    //TODO: via settings determine order of processing (if device could overlap kernel executing and memory copying, use this way)
    template<class SceneType>
    void process_buffer(const SceneType* scene_ptr)
    {
        WHIRL_CHECK_NOT_EQ(scene_ptr, nullptr);
        
        size_t alloc_width = chunk_buffer_->get_alloc_size().cx;

        auto memory_operation = [this, scene_ptr, alloc_width](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            pinned_mem_manager_->async_copy_to_device(device_buffer_.get_allocated_chunk(index), 
                                                      render_buffer_.get_allocated_chunk(index), height_size*alloc_width, stream_handle);

            wh_launch_kernel<SceneType>(device_buffer_, device_buffer_.get_allocated_chunk(index), scene_ptr, height_shift, height_size, stream_handle);

            pinned_mem_manager_->async_copy_from_device(render_buffer_.get_allocated_chunk(index), 
                                                        device_buffer_.get_allocated_chunk(index), height_size*alloc_width, stream_handle);
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization
        
        WHAbstractMemoryStrategy::process_buffer<SceneType>(scene_ptr);
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
    WHManagedMemoryStrategy(const std::shared_ptr<WHEvent>& synchronizer, const WHBuffer::Size_& buf_size, int device_set, unsigned int alloc_flags = 0u, unsigned int attach_flags = cudaMemAttachSingle):
        managed_mem_manager_(WHMemoryManager<WHAllocType::MANAGED>::instance()), device_(device_set),
        WHAbstractMemoryStrategy(synchronizer, std::dynamic_pointer_cast<WHAbstractMemoryManager>(managed_mem_manager_),
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
    template<class SceneType>
    void process_buffer(const SceneType* scene_ptr)
    {
        WHIRL_CHECK_NOT_EQ(scene_ptr, nullptr);
        
        size_t alloc_width = chunk_buffer_->get_alloc_size().cx;

        auto memory_operation = [this, scene_ptr, alloc_width](size_t index, size_t height_shift, size_t height_size, cudaStream_t stream_handle)
        {
            managed_mem_manager_->prefetch_to_device(render_buffer_.get_allocated_chunk(index), height_size*alloc_width, device_, stream_handle);
            wh_launch_kernel<SceneType>(device_buffer_, render_buffer_.get_allocated_chunk(index), scene_ptr, height_shift, height_size, stream_handle);
            managed_mem_manager_->prefetch_to_host(render_buffer_.get_allocated_chunk(index), height_size*alloc_width, stream_handle);
        };

        chunk_buffer_->iterate(memory_operation);//must cause lambda - optimization
            
        WHAbstractMemoryStrategy::process_buffer<SceneType>(scene_ptr);//mb to call it after synchronize
    
        WHIRL_CUDA_CALL(cudaDeviceSynchronize());
    }

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::MANAGED>> managed_mem_manager_;
    
    int device_;
};

}//namespace whirl