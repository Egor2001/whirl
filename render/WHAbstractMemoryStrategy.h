#pragma once

#include <memory>
#include <vector>

#include "WHAbstractBuffer.h"
#include "WHStream.h"
#include "WHMemoryManager.h"
#include "WHStreamManager.h"

namespace whirl {

class WHAbstractMemoryStrategy abstract
{
public:
    WHAbstractMemoryStrategy() = default;
    WHAbstractMemoryStrategy(const WHBuffer& render_buffer_set): render_buffer_(render_buffer_set) {}

    virtual ~WHAbstractMemoryStrategy() = default;

    virtual void process_buffer(/* pointer to scene structure and other parameters */) = 0;
    
          WHBuffer& get_buffer()       { return render_buffer_; }//TODO: add synchronization via events or callbacks
    const WHBuffer& get_buffer() const { return render_buffer_; }//TODO: add synchronization via events or callbacks
        
protected:
    WHBuffer render_buffer_;
};

class WHSimpleMemoryStrategy : public WHAbstractMemoryStrategy
{
public:
    WHSimpleMemoryStrategy(): 
        host_mem_manager_  (WHMemoryManager<WHAllocType::HOST>  ::instance()),
        device_mem_manager_(WHMemoryManager<WHAllocType::DEVICE>::instance()),
        device_buffer_{}, WHAbstractMemoryStrategy(WHBuffer{}) {}

    virtual ~WHSimpleMemoryStrategy() override = default;

private:
    std::shared_ptr<WHMemoryManager<WHAllocType::HOST>>   host_mem_manager_;
    std::shared_ptr<WHMemoryManager<WHAllocType::DEVICE>> device_mem_manager_;
    
    WHBuffer device_buffer_;
};

}//namespace whirl