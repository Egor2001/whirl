#pragma once

#include <vector>
#include <type_traits>
#include <thread>
#include <stdexcept>

#include "../logger/WHLogger.h"
#include "WHStream.h"
#include "WHMemoryChunk.h"

namespace whirl {

//TODO: singleton
class WHStreamManager
{
private:
    WHStreamManager(): WHStreamManager(((MAX_STREAMS_ + 1u)*3)/4) {}
    
    explicit WHStreamManager(uint16_t engine_streams_num): engine_stream_vec_{}, special_stream_vec_{}
    {
        WHIRL_TRACE       ("creating stream manager [engine streams num: {0:d}]", engine_streams_num);
        WHIRL_CHECK_NOT_EQ(engine_streams_num, 0u);
        
        engine_stream_vec_.resize(engine_streams_num);
        
        for (auto& it : engine_stream_vec_) 
            it = std::make_shared<WHStream>(cudaStreamDefault);        
    }

public:
    static std::shared_ptr<WHStreamManager> instance() { return instance_; }

    ~WHStreamManager()
    {
        WHIRL_TRACE("releasing stream manager");

        engine_stream_vec_ .clear();
        special_stream_vec_.clear();
    }
    
    std::shared_ptr<WHChunkBuffer> get_chunks(const WHChunkBuffer::Size_& pixel_size)
    {
        return std::make_shared<WHChunkBuffer>(pixel_size, engine_stream_vec_);
    }

    std::shared_ptr<WHStream> get_stream() 
    {
        auto iter = std::find_if(special_stream_vec_.begin(), special_stream_vec_.end(), 
                                 [](const std::shared_ptr<WHStream>& stream_ptr) { return stream_ptr->query(); });

        if (iter != special_stream_vec_.end())
        {
            WHIRL_TRACE("stream manager get stream [[IDLE], handle: {0:d}]", (*iter)->get_stream_handle());

            return *iter;
        }
        
        if (special_stream_vec_.size() + engine_stream_vec_.size() < MAX_STREAMS_)
        {
            iter = add_stream_();
            
            WHIRL_TRACE("stream manager get stream [[ADDED], handle: {0:d}]", (*iter)->get_stream_handle());

            return *iter;
        }
        
        iter = std::min_element(special_stream_vec_.begin(), special_stream_vec_.end(), 
                                [](const std::shared_ptr<WHStream>& stream_ptr) { return stream_ptr.use_count(); });

        WHIRL_TRACE("stream manager get stream [[DEFAULT], handle: {0:d}]", (*iter)->get_stream_handle());

        return *iter;
    }

private:
    template<typename... Types>
    std::vector<std::shared_ptr<WHStream>>::iterator add_stream_(Types&&... args)
    {
        special_stream_vec_.push_back(std::make_shared<WHStream>(std::forward<Types>(args)...));//TODO: to create  class WHStreamExt derived from WHStream with cuda libraries handles
        
        WHIRL_TRACE("add stream to stream manager [handle: {0:p}]", special_stream_vec_.back()->get_stream_handle());

        return special_stream_vec_.end()-1;
    }

private:
    static uint16_t MAX_STREAMS_;

    std::vector<std::shared_ptr<WHStream>> engine_stream_vec_, special_stream_vec_;

private:
    static std::shared_ptr<WHStreamManager> instance_;    
};

uint16_t WHStreamManager::MAX_STREAMS_ = strtoul(getenv("CUDA_DEVICE_MAX_CONNECTIONS"), nullptr, 0) - 1u;

std::shared_ptr<WHStreamManager> instance_ = std::make_shared<WHStreamManager>();

}//namespace whirl