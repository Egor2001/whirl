#pragma once

#include <vector>
#include <type_traits>

#include "WHStream.h"

namespace whirl {

class WHMemoryChunk
{
public:
    WHMemoryChunk() = default;
    WHMemoryChunk(void* src_ptr_set, void* dst_ptr_set, size_t size_set, const WHStream* stream_set = nullptr): 
        src_ptr_(src_ptr_set), dst_ptr_(dst_ptr_set), size_(size_set), stream_(stream_set) {}

    ~WHMemoryChunk()
    {
        src_ptr_ = dst_ptr_ = nullptr;
        size_ = 0;
        stream_ = nullptr;
    }

private:
    void* src_ptr_;
    void* dst_ptr_;
    size_t size_;
    
    const WHStream* stream_;
};

//TODO: singleton
class WHStreamManager
{
public:
    WHStreamManager()
    {
        for (size_t i = 0; i < 4; i++)
            stream_vec_.push_back(std::make_shared<WHStream>(cudaStreamDefault));
    }
    
    ~WHStreamManager()
    {
        stream_vec_.clear();
    }

    //Just very simple implementation
    template<typename Type_t>
    std::vector<WHMemoryChunk> decompose(Type_t* src_ptr, Type_t* dst_ptr, size_t size)
    {
        size_t idle_streams_num = stream_vec_.size();//TODO: count only idle streams
        
        std::vector<WHMemoryChunk> result = std::vector<WHMemoryChunk>(idle_streams_num);
        
        for (size_t i = 0; i < idle_streams_num; i++)
        {
            float delta = float(size)/idle_streams_num;

            result[i] = WHMemoryChunk(src_ptr + ptrdiff_t(i*delta), dst_ptr + ptrdiff_t(i*delta), 
                                      ptrdiff_t((i+1)*delta) - ptrdiff_t(i*delta), stream_vec_[i]);
        }
        
        return result;
    }

private:
    std::vector<std::shared_ptr<WHStream>> stream_vec_;
};

}//namespace whirl