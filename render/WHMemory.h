#pragma once

#include <stdexcept>
#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"
#include "WHStream.h"

namespace whirl {

enum class WHMemoryLocation
{
    CPU = 0, GPU = 1
};

enum class WHAllocType
{
    NONE = 0, HOST = 1, PINNED = 2, DEVICE = 3, MANAGED = 4 
};

class WHBaseMemoryManager abstract
{
public:
    WHBaseMemoryManager() = delete;
    WHBaseMemoryManager(WHAllocType alloc_type_set): alloc_type_(alloc_type_set) {}

    virtual ~WHBaseMemoryManager() { alloc_type_ = WHAllocType::NONE; }

    virtual void* allocate  (size_t size, unsigned int flags = 0) const = 0;
    virtual void  deallocate(void* ptr)                           const = 0;
    
    virtual void memory_set      (void* dst, int value,       size_t size, cudaStream_t stream = nullptr) const = 0;
    virtual void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const = 0;
    virtual void copy_to_device  (void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const = 0;

    WHAllocType alloc_type() const { return alloc_type_; }
    
private:
    WHAllocType alloc_type_;
};

template<class MemoryManager_>
class WHSingletonMemoryManager abstract : public WHBaseMemoryManager
{
protected:
    WHSingletonMemoryManager(WHAllocType alloc_type_set): WHBaseMemoryManager(alloc_type_set) {}

private:
    WHSingletonMemoryManager() = delete;
    
    WHSingletonMemoryManager             (const WHSingletonMemoryManager&) = delete;
    WHSingletonMemoryManager& operator = (const WHSingletonMemoryManager&) = delete;

    WHSingletonMemoryManager             (WHSingletonMemoryManager&&) = delete;
    WHSingletonMemoryManager& operator = (WHSingletonMemoryManager&&) = delete;
    
public:
    virtual ~WHSingletonMemoryManager() = default;
    
    virtual void* allocate  (size_t size, unsigned int flags = 0) const override = 0;
    virtual void  deallocate(void* ptr)                           const override = 0;
    
    virtual void memory_set      (void* dst, int value,       size_t size, cudaStream_t stream = nullptr) const override = 0;
    virtual void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const override = 0;
    virtual void copy_to_device  (void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const override = 0;

    static std::shared_ptr<WHBaseMemoryManager> instance()
    {
        if (!instance_) instance_ = std::shared_ptr<WHBaseMemoryManager>(new MemoryManager_);

        return instance_;
    }

private:
    static std::shared_ptr<WHBaseMemoryManager> instance_;
};

template<class MemoryManager_>
std::shared_ptr<WHBaseMemoryManager> WHSingletonMemoryManager<MemoryManager_>::instance_;

template<WHAllocType> class WHHostMemoryManager;

template<>
class WHHostMemoryManager<WHAllocType::HOST> : public WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::HOST>>
{
private:
    friend class WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::HOST>>;

    WHHostMemoryManager(): WHSingletonMemoryManager(WHAllocType::HOST) {}

public:
    virtual ~WHHostMemoryManager() = default;

    virtual void* allocate(size_t size, unsigned int flags = 0) const override 
    {
        void* result = std::malloc(size);

        WHIRL_TRACE("alloc host [pointer: host_{0:p}, size: {1}] {2}", result, size, (result ? "success" : " FAILURE"));

        return result;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_TRACE("dealloc host [pointer: host_{0:p}]", ptr);
        std::free(ptr);
    }
    
    virtual void memory_set(void* dst, int value, size_t size, cudaStream_t stream = nullptr) const override
    {
        WHIRL_TRACE("memset host [pointer: host_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        std::memset(dst, value, size);
    }
   
    virtual void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const override
    {
        WHIRL_TRACE    ("memcpy from device to host [destination: host_{0:p}, source: device_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
    
    virtual void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const override
    {
        WHIRL_TRACE    ("memcpy from host to device [destination: device_{0:p}, source: host_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
};

template<>
class WHHostMemoryManager<WHAllocType::DEVICE> : public WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::DEVICE>>
{
private:
    friend class WHSingletonMemoryManager<WHHostMemoryManager>;

    WHHostMemoryManager(): WHSingletonMemoryManager(WHAllocType::DEVICE) {}

public:
    virtual ~WHHostMemoryManager() = default;

    virtual void* allocate(size_t size, unsigned int flags = 0) const override
    {
        void* result = nullptr;

        WHIRL_CUDA_CALL(cudaMalloc(&result, size));
        WHIRL_TRACE    ("alloc device [pointer: device_{0:p}, size: {1}] {2}", result, size, (result ? "success" : " FAILURE"));

        return result;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_TRACE    ("dealloc device [pointer: device_{0:p}]", ptr);
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    virtual void memory_set(void* dst, int value, size_t size, cudaStream_t stream = nullptr) const override
    {
        WHIRL_TRACE    ("memset device [pointer: device_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        WHIRL_CUDA_CALL(cudaMemset(dst, value, size));
    }

    virtual void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const override
    {
        WHIRL_TRACE    ("memcpy from device to device [destination: device_{0:p}, source: device_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
    
    virtual void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const override
    {
        WHIRL_TRACE    ("memcpy from device to device [destination: device_{0:p}, source: device_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
};

template<>
class WHHostMemoryManager<WHAllocType::PINNED> : public WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::PINNED>>
{
private:
    friend class WHSingletonMemoryManager<WHHostMemoryManager>;

    WHHostMemoryManager(): WHSingletonMemoryManager(WHAllocType::PINNED) {}

public:
    virtual ~WHHostMemoryManager() = default;

    virtual void* allocate(size_t size, unsigned int flags = 0) const override
    {
        void* result = nullptr;

        WHIRL_CUDA_CALL(cudaHostAlloc(&result, size, flags));
        WHIRL_TRACE    ("alloc pinned [pointer: pinned_{0:p}, size: {1}] {2}", result, size, (result ? "success" : " FAILURE"));

        return result;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_TRACE    ("dealloc pinned [pointer: pinned_{0:p}]", ptr);
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    virtual void memory_set(void* dst, int value, size_t size, cudaStream_t stream) const override
    {
        WHIRL_TRACE    ("memset pinned [pointer: pinned_{0:p}, size: {1}, value: {2:#x}, stream_handle: {3:#x}]", dst, size, value, stream);
        WHIRL_CUDA_CALL(cudaMemsetAsync(dst, value, size, stream));
    }

    virtual void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream) const override
    {
        WHIRL_TRACE("memcpy from device to pinned [destination: pinned_{0:p}, source: device_{1:p}, "
                                                  "size: {2}, stream_handle: {3:#x}]", dst, src, size, stream);
        WHIRL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    }
    
    virtual void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) const override
    {
        WHIRL_TRACE("memcpy from pinned to device [destination: device_{0:p}, source: pinned_{1:p}, "
                                                   "size: {2}, stream_handle: {3:#x}]", dst, src, size, stream);
        WHIRL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    }
};

}//namespace whirl