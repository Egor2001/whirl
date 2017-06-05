#pragma once

#include <memory>
#include <vector>

#include "WHAbstractBuffer.h"
#include "WHStream.h"
#include "WHMemoryManager.h"

namespace whirl {

class WHAbstractMemoryStrategy abstract
{
public:
    WHAbstractMemoryStrategy(WHAllocType alloc_type_set): alloc_type_(alloc_type_set) {}
    
    virtual ~WHAbstractMemoryStrategy() { alloc_type_ = WHAllocType::NONE; }

    WHAllocType get_alloc_type() { return alloc_type_; }

    virtual void                              process_buffer(, const std::vector<std::shared_ptr<WHStream>>&) = 0;
    virtual std::shared_ptr<WHAbstractBuffer> get_buffer    () = 0;
    
protected:
    WHAllocType alloc_type_;
};

}//namespace whirl