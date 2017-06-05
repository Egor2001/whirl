#pragma once

#include <algorithm>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"

namespace whirl {

class WHStream
{
public:
    WHStream():                   stream_handle_(nullptr) {}
    WHStream(unsigned int flags): stream_handle_(nullptr)
    {
        WHIRL_CUDA_CALL(cudaStreamCreateWithFlags(&stream_handle_, flags));
        WHIRL_TRACE("stream created [handle: {0:p}]", stream_handle_);
    }

    WHStream             (const WHStream&) = delete;
    WHStream& operator = (const WHStream&) = delete;
   
    WHStream(WHStream&& src_stream): stream_handle_(src_stream.stream_handle_)
    {
        src_stream.stream_handle_ = nullptr;
    }

    WHStream& operator = (WHStream&& src_stream)
    {
        stream_handle_            = src_stream.stream_handle_;
        src_stream.stream_handle_ = nullptr;
    }

    virtual ~WHStream()
    {
        if(stream_handle_)
            WHIRL_CUDA_CALL(cudaStreamDestroy(stream_handle_));

        WHIRL_TRACE("stream destroyed [handle: {0:p}]", stream_handle_);
    
        stream_handle_ = nullptr;
    }

    void reset(unsigned int flags)
    {
        if(stream_handle_)
            WHIRL_CUDA_CALL(cudaStreamDestroy(stream_handle_));
        
        cudaStream_t old_handle = stream_handle_; stream_handle_ = nullptr;

        WHIRL_CUDA_CALL(cudaStreamCreateWithFlags(&stream_handle_, flags));
        WHIRL_TRACE("stream reset [old handle: {0:p}, new handle: {1:p}]", old_handle, stream_handle_);
    }

    cudaStream_t get_stream_handle() const noexcept { return stream_handle_; }

private:
    cudaStream_t stream_handle_;
};

}//namespace whirl