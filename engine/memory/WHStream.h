#pragma once

#include <functional>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"

namespace whirl {

class WHStream
{
public:
    WHStream(unsigned int flags = cudaStreamDefault) noexcept : stream_handle_(nullptr)
    {
        WHIRL_CUDA_CALL(cudaStreamCreateWithFlags(&stream_handle_, flags));
        WHIRL_TRACE    ("stream created [handle: {0:p}]", stream_handle_);
    }

    WHStream             (const WHStream&) = delete;
    WHStream& operator = (const WHStream&) = delete;
   
    WHStream(WHStream&& src_stream) noexcept : stream_handle_(src_stream.stream_handle_)
    {
        src_stream.stream_handle_ = nullptr;
    }

    WHStream& operator = (WHStream&& src_stream) noexcept
    {
        stream_handle_            = src_stream.stream_handle_;
        src_stream.stream_handle_ = nullptr;
    }

    virtual ~WHStream()
    {
        WHIRL_TRACE("stream destroyed [handle: {0:p}]", stream_handle_);
    
        synchronize();

        if(stream_handle_)
            WHIRL_CUDA_CALL(cudaStreamDestroy(stream_handle_));
                
        stream_handle_ = nullptr;
    }

    void reset(unsigned int flags = cudaStreamDefault)
    {
        if (stream_handle_) 
            WHIRL_CUDA_CALL(cudaStreamDestroy(stream_handle_));
        
        cudaStream_t old_handle = stream_handle_; stream_handle_ = nullptr;

        WHIRL_CUDA_CALL(cudaStreamCreateWithFlags(&stream_handle_, flags));
        WHIRL_TRACE    ("stream reset [old handle: {0:p}, new handle: {1:p}]", old_handle, stream_handle_);
    }

    bool query() const noexcept 
    {
        WHIRL_TRACE("stream query [handle: {0:p}]", stream_handle_);
    
        return (WHIRL_CUDA_CALL(cudaStreamQuery(stream_handle_)) == cudaSuccess);
    }

    void synchronize() const noexcept 
    {
        WHIRL_TRACE    ("stream synchronize [handle: {0:p}]", stream_handle_);
        WHIRL_CUDA_CALL(cudaStreamSynchronize(stream_handle_));
    }
    
    void wait_event(cudaEvent_t cuda_event, unsigned int flags) const noexcept 
    {
        WHIRL_TRACE    ("stream wait event [handle: {0:p}, event: {1:p}, flags: {2:#x}]", stream_handle_, cuda_event, flags);
        WHIRL_CUDA_CALL(cudaStreamWaitEvent(stream_handle_, cuda_event, flags));
    }
    
    void attach_mem_async(void* dev_ptr, unsigned int flags = cudaMemAttachSingle) const noexcept 
    {
        WHIRL_TRACE    ("stream attach mem async [handle: {0:p}, pointer: {1:p}, flags: {2:#x}]", stream_handle_, dev_ptr, flags);
        WHIRL_CUDA_CALL(cudaStreamAttachMemAsync(stream_handle_, dev_ptr, 0, flags));
    }

    template<typename... ArgTypes>
    void add_callback(std::function<void CUDART_CB (cudaStream_t, cudaError_t, void*, ArgTypes&&...)> func, ArgTypes&&... args) const noexcept 
    {
        WHIRL_TRACE    ("stream add callback [handle: {0:p}, function: {1:s}]", stream_handle_, typeid(func).name());
        WHIRL_CUDA_CALL(cudaStreamAddCallback(stream_handle_, 
                        std::bind(func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::forward<ArgTypes>(args)...), 
                        static_cast<void*>(this), 0));
    }

    cudaStream_t get_stream_handle() const noexcept { return stream_handle_; }

protected:
    cudaStream_t stream_handle_;
};

}//namespace whirl