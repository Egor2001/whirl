#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"

namespace whirl {

class WHEvent final
{
public:
    WHEvent(unsigned int flags = cudaEventDefault) noexcept : event_handle_{nullptr}
    {
        WHIRL_CUDA_CALL(cudaEventCreateWithFlags(&event_handle_, flags));
    }
    
    WHEvent             (const WHEvent&) = delete;
    WHEvent& operator = (const WHEvent&) = delete;
    
    WHEvent(WHEvent&& move_event) noexcept : event_handle_(move_event.event_handle_) 
    {
        move_event.event_handle_ = nullptr;
    }

    WHEvent& operator = (WHEvent&& move_event) noexcept
    {
        event_handle_ = move_event.event_handle_;
        move_event.event_handle_ = nullptr;
    }

    ~WHEvent()
    {
        WHIRL_CUDA_CALL(cudaEventDestroy(event_handle_));
        event_handle_ = nullptr;
    }

    void reset(unsigned int flags = cudaEventDefault)
    {
        if (event_handle_)
            WHIRL_CUDA_CALL(cudaEventDestroy(event_handle_));

        WHIRL_CUDA_CALL(cudaEventCreateWithFlags(&event_handle_, flags));
    }

    void record(cudaStream_t stream = nullptr) const noexcept 
    {
        WHIRL_CUDA_CALL(cudaEventRecord(event_handle_, stream));
    }
        
    bool query() const noexcept 
    {
        return WHIRL_CUDA_CALL(cudaEventQuery(event_handle_)) == cudaSuccess;
    }

    void synchronize() const noexcept 
    {
        WHIRL_CUDA_CALL(cudaEventSynchronize(event_handle_));
    }

    float elapsed_time(cudaEvent_t end_event) const noexcept 
    {
        float result = 0.0f;

        WHIRL_CUDA_CALL(cudaEventElapsedTime(&result, event_handle_, end_event));

        return result;
    }

    cudaEvent_t get_event_handle() const noexcept { return event_handle_; }

private:
    cudaEvent_t event_handle_;
};

}//namespace whirl