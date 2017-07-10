#pragma once

#include <stdexcept>
#include <typeinfo>
#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"

#include "../logger/WHLogger.h"
#include "WHStream.h"

namespace whirl {

enum class WHAllocType
{
    NONE = 0, HOST = 1, PINNED = 2, MANAGED = 3, DEVICE = 4 
};

class WHAbstractMemoryManager abstract
{
public:
    WHAbstractMemoryManager():                           alloc_type_(WHAllocType::NONE) {}
    WHAbstractMemoryManager(WHAllocType alloc_type_set): alloc_type_(alloc_type_set)    {}

    virtual ~WHAbstractMemoryManager()
    {
        alloc_type_ = WHAllocType::NONE;
    }

    WHAllocType alloc_type() const { return alloc_type_; }

    virtual void* allocate  (size_t size, unsigned int flags) const = 0;
    virtual void  deallocate(void* ptr)                       const = 0;

protected:
    WHAllocType alloc_type_;
};

template<class MemoryManager_>
class WHSingletonMemoryManager abstract : public WHAbstractMemoryManager
{
protected:
    WHSingletonMemoryManager(WHAllocType alloc_type_set): WHAbstractMemoryManager(alloc_type_set) 
    {
        WHIRL_TRACE("creating singleton memory manager [class name: {0:s}, alloc type: {1:d}]", typeid(MemoryManager_).name(), alloc_type_);
        
        if (!MemoryManager_::is_supported(device))//TODO:<----
            WHIRL_DEBUG("memory manager [{:s}] is not supported by chosen device", typeid(MemoryManager_).name());
    }

private:
    WHSingletonMemoryManager() = delete;
    
    WHSingletonMemoryManager             (const WHSingletonMemoryManager&) = delete;
    WHSingletonMemoryManager& operator = (const WHSingletonMemoryManager&) = delete;

    WHSingletonMemoryManager             (WHSingletonMemoryManager&&) = delete;
    WHSingletonMemoryManager& operator = (WHSingletonMemoryManager&&) = delete;
    
public:
    virtual ~WHSingletonMemoryManager() override
    {
        WHIRL_TRACE("releasing singleton memory manager [class name: {0:s}, alloc type: {1:d}]", typeid(MemoryManager_).name(), alloc_type_);
    }
    
    virtual void* allocate  (size_t size, unsigned int flags) const override = 0;
    virtual void  deallocate(void* ptr)                       const override = 0;

    static std::shared_ptr<MemoryManager_> instance()
    {
        if (!instance_) instance_ = std::shared_ptr<MemoryManager_>(new MemoryManager_);

        return instance_;
    }

private:
    static std::shared_ptr<MemoryManager_> instance_;
};

template<class MemoryManager_>
std::shared_ptr<MemoryManager_> WHSingletonMemoryManager<MemoryManager_>::instance_;

template<WHAllocType> class WHMemoryManager;

template<>
class WHMemoryManager<WHAllocType::HOST> : public WHSingletonMemoryManager<WHMemoryManager<WHAllocType::HOST>>
{
private:
    friend class WHSingletonMemoryManager<WHMemoryManager<WHAllocType::HOST>>;

    WHMemoryManager(): WHSingletonMemoryManager(WHAllocType::HOST) {}

public:
    virtual ~WHMemoryManager() override = default;

    static bool is_supported(int device)
    { 
        return true; 
    }

    virtual void* allocate(size_t size, unsigned int flags = 0u) const override
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
    
    void memory_set(void* dst, int value, size_t size) const
    {
        WHIRL_TRACE("memset host [pointer: host_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        std::memset(dst, value, size);
    }
   
    void copy_from_device(void* dst, const void* src, size_t size) const
    {
        WHIRL_TRACE    ("memcpy from device to host [destination: host_{0:p}, source: device_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
    
    void copy_to_device(void* dst, const void* src, size_t size) const
    {
        WHIRL_TRACE    ("memcpy from host to device [destination: device_{0:p}, source: host_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
};

template<>
class WHMemoryManager<WHAllocType::PINNED> : public WHSingletonMemoryManager<WHMemoryManager<WHAllocType::PINNED>>
{
private:
    friend class WHSingletonMemoryManager<WHMemoryManager>;

    WHMemoryManager(): WHSingletonMemoryManager(WHAllocType::PINNED) {}

public:
    virtual ~WHMemoryManager() override = default;

    static bool is_supported(int device) 
    { 
        return true; 
    }

    virtual void* allocate(size_t size, unsigned int flags = cudaHostAllocDefault) const override
    {
        void* result = nullptr;

        WHIRL_CUDA_CALL(cudaHostAlloc(&result, size, flags));
        WHIRL_TRACE    ("alloc pinned [pointer: pinned_{0:p}, size: {1}, flags: {2:d}] {3}", result, size, flags, (result ? "success" : " FAILURE"));

        return result;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_TRACE    ("dealloc pinned [pointer: pinned_{0:p}]", ptr);
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    void async_memory_set(void* dst, int value, size_t size, cudaStream_t stream) const
    {
        WHIRL_TRACE    ("memset pinned [pointer: pinned_{0:p}, size: {1}, value: {2:#x}, stream_handle: {3:#x}]", dst, size, value, stream);
        WHIRL_CUDA_CALL(cudaMemsetAsync(dst, value, size, stream));
    }

    void async_copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream) const
    {
        WHIRL_TRACE("memcpy from device to pinned [destination: pinned_{0:p}, source: device_{1:p}, "
                                                  "size: {2}, stream_handle: {3:#x}]", dst, src, size, stream);
        WHIRL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    }
    
    void async_copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) const
    {
        WHIRL_TRACE("memcpy from pinned to device [destination: device_{0:p}, source: pinned_{1:p}, "
                                                   "size: {2}, stream_handle: {3:#x}]", dst, src, size, stream);
        WHIRL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    }
};

//TODO: use this memory manager if concurrentManagedAccess is not 0 for specified device. 
//So may be it's need to allocate managed memory only with cudaAttachHost flag 
template<>
class WHMemoryManager<WHAllocType::MANAGED> : public WHSingletonMemoryManager<WHMemoryManager<WHAllocType::MANAGED>>
{
private:
    friend class WHSingletonMemoryManager<WHMemoryManager>;

    WHMemoryManager(): WHSingletonMemoryManager(WHAllocType::MANAGED) {}

public:
    virtual ~WHMemoryManager() override = default;
    
    static bool is_supported(int device) 
    {
        int result = 0;

        cudaDeviceGetAttribute(&result, cudaDevAttrManagedMemory, device);

        return !!result; 
    }

    virtual void* allocate(size_t size, unsigned int flags = cudaMemAttachGlobal) const override
    {
        void* result = nullptr;

        WHIRL_TRACE    ("alloc managed [pointer: device_{0:p}, size: {1}, flags: {2:#x}]", result, size, flags);
        WHIRL_CUDA_CALL(cudaMallocManaged(&result, size, flags));
        
        return result;
    }

    virtual void deallocate(void* ptr) const override
    {
        WHIRL_TRACE    ("dealloc managed [pointer: managed_{0:p}]", ptr);
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    void memory_set(void* dst, int value, size_t size) const
    {
        WHIRL_TRACE    ("memset managed [pointer: managed_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        WHIRL_CUDA_CALL(cudaMemset(dst, value, size));
    }

    void prefetch_to_host(const void* ptr, size_t size, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE    ("prefetch to host [pointer: managed_{0:p}, size: {1}, device: cudaCpuDeviceId, stream: {2:#x}]", ptr, size, stream);
        WHIRL_CUDA_CALL(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream));
    }
    
    void prefetch_to_device(const void* ptr, size_t size, int device, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE    ("prefetch to device [pointer: managed_{0:p}, size: {1}, device: {2:d}, stream: {3:#x}]", ptr, size, device, stream);
        WHIRL_CUDA_CALL(cudaMemPrefetchAsync(ptr, size, device, stream));
    }

    //TODO: always use while executing kernel in stream (allows to access associated memory from host while device in this stream is idle)
    void attach_async(cudaStream_t stream, const void* ptr, unsigned int flags = cudaMemAttachSingle)
    {
        WHIRL_TRACE    ("attach async [pointer: managed_{0:p}, stream: {1:#x}, flags: {2:#x}]", ptr, stream, flags);
        WHIRL_CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr, 0, flags));
    }

    void mem_advice(const void* ptr, size_t size, cudaMemoryAdvise advice, int device = cudaCpuDeviceId)
    {
        WHIRL_TRACE    ("memory advice [pointer: managed_{0:p}, size: {1}, advice: {2:d}, device: {3:d}]", ptr, size, advice, device);
        WHIRL_CUDA_CALL(cudaMemAdvise(ptr, size, advice, device));
    }
};

template<>
class WHMemoryManager<WHAllocType::DEVICE> : public WHSingletonMemoryManager<WHMemoryManager<WHAllocType::DEVICE>>
{
private:
    friend class WHSingletonMemoryManager<WHMemoryManager>;

    WHMemoryManager(): WHSingletonMemoryManager(WHAllocType::DEVICE) {}

public:
    virtual ~WHMemoryManager() override = default;
    
    static bool is_supported(int device) 
    {
        return true;
    }

    virtual void* allocate(size_t size, unsigned int flags = 0u) const override
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
    
    void memory_set(void* dst, int value, size_t size) const
    {
        WHIRL_TRACE    ("memset device [pointer: device_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        WHIRL_CUDA_CALL(cudaMemset(dst, value, size));
    }
};

}//namespace whirl