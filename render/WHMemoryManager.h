#pragma once

#include <stdexcept>
#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"

enum WHMemoryLocation
{
    CPU = 0, GPU = 1
};

enum class WHAllocType
{
    NONE = 0, CPU = 1, GPU = 2, HOST = 3 
};
//TODO: create shared_ptr
class WHMemoryManager abstract
{
public:
    WHMemoryManager() = delete;
    WHMemoryManager(WHAllocType alloc_type_set): alloc_type_(alloc_type_set) {}

    virtual ~WHMemoryManager() { alloc_type_ = WHAllocType::NONE; }

    virtual void* allocate  (size_t size) const = 0;
    virtual void  deallocate(void* ptr)   const = 0;
    
    virtual void memory_set (void* dst, int value, size_t size) const = 0;
    virtual void memory_copy(void* dst, const void* src, WHAllocType src_alloc_type, size_t size) const = 0;

    WHAllocType alloc_type() const { return alloc_type_; }
    
private:
    WHAllocType alloc_type_;
};

class WHCpuMemManager : public WHMemoryManager
{
public:
    WHCpuMemManager(): WHMemoryManager(WHAllocType::CPU) {}
    virtual ~WHCpuMemManager() = default;

    virtual void* allocate(size_t size) const override
    {
        return std::malloc(size);
    }

    virtual void deallocate(void* ptr) const override
    {
        std::free(ptr);
    }
    
    virtual void memory_set(void* dst, int value, size_t size) const override
    {
        std::memset(dst, value, size);
    }

    virtual void memory_copy(void* dst, const void* src, WHAllocType src_alloc_type, size_t size) const override
    {
        switch (src_alloc_type)//TODO: throw if none
        {
        case WHAllocType::NONE: throw std::logic_error("source alloc type is none"); break;
        
        case WHAllocType::CPU:                  std::memcpy(dst, src, size);                          break;
        case WHAllocType::GPU:  WHIRL_CUDA_CALL(cudaMemcpy (dst, src, size, cudaMemcpyDeviceToHost)); break;
        case WHAllocType::HOST: WHIRL_CUDA_CALL(cudaMemcpy (dst, src, size, cudaMemcpyDeviceToHost)); break;//TODO: cudaMemcpyAsync

        default: throw std::logic_error("source other alloc type"); break;
        }
    }
};

class WHGpuMemManager : public WHMemoryManager
{
public:
    WHGpuMemManager(): WHMemoryManager(WHAllocType::GPU) {}
    virtual ~WHGpuMemManager() = default;

    virtual void* allocate(size_t size) const override
    {
        void* ptr = nullptr;

        WHIRL_CUDA_CALL(cudaMalloc(&ptr, size));
        
        return ptr;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    virtual void memory_set(void* dst, int value, size_t size) const override
    {
        WHIRL_CUDA_CALL(cudaMemset(dst, value, size));
    }

    virtual void memory_copy(void* dst, const void* src, WHAllocType src_alloc_type, size_t size) const override
    {
        switch (src_alloc_type)//TODO: throw if none
        {
        case WHAllocType::NONE: throw std::logic_error("source alloc type is none"); break;
        
        case WHAllocType::CPU:  WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));   break;
        case WHAllocType::GPU:  WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)); break;
        case WHAllocType::HOST: WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)); break;

        default: throw std::logic_error("invalid source alloc type"); break;
        }
    }
};
//TODO create async memory manager
class WHHostMemManager : public WHMemoryManager
{
public:
    WHHostMemManager(): WHMemoryManager(WHAllocType::GPU) {}
    virtual ~WHHostMemManager() = default;

    virtual void* allocate(size_t size) const override
    {
        void* ptr = nullptr;

        WHIRL_CUDA_CALL(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
        
        return ptr;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_CUDA_CALL(cudaFreeHost(ptr));
    }
    
    virtual void memory_set(void* dst, int value, size_t size) const override
    {
        WHIRL_CUDA_CALL(cudaMemset(dst, value, size));
    }

    virtual void memory_copy(void* dst, const void* src, WHAllocType src_alloc_type, size_t size) const override
    {
        switch (src_alloc_type)//TODO: throw if none
        {
        case WHAllocType::NONE: throw std::logic_error("source alloc type is none"); break;
        
        case WHAllocType::CPU:  WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));   break;
        case WHAllocType::GPU:  WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)); break;
        case WHAllocType::HOST: WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)); break;//TODO: cudaMemcpyAsync

        default: throw std::logic_error("invalid source alloc type"); break;
        }
    }
};
