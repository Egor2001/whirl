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

template<class MemoryManager_>
class WHSingletonMemoryManager abstract
{
protected:
    WHSingletonMemoryManager(WHAllocType alloc_type_set): alloc_type_(alloc_type_set) 
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
    virtual ~WHSingletonMemoryManager()
    {
        WHIRL_TRACE("releasing singleton memory manager [class name: {0:s}, alloc type: {1:d}]", typeid(MemoryManager_).name(), alloc_type_);

        alloc_type_ = WHAllocType::NONE;
    }
    
    static std::shared_ptr<MemoryManager_> instance()
    {
        if (!instance_) instance_ = std::shared_ptr<MemoryManager_>(new MemoryManager_);

        return instance_;
    }

private:
    static std::shared_ptr<MemoryManager_> instance_;

    WHAllocType alloc_type_;
};

template<class MemoryManager_>
std::shared_ptr<MemoryManager_> WHSingletonMemoryManager<MemoryManager_>::instance_;

template<WHAllocType> class WHHostMemoryManager;

template<>
class WHHostMemoryManager<WHAllocType::HOST> : public WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::HOST>>
{
private:
    friend class WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::HOST>>;

    WHHostMemoryManager(): WHSingletonMemoryManager(WHAllocType::HOST) {}

public:
    virtual ~WHHostMemoryManager() = default;

    static bool is_supported(int device) 
    { 
        return true; 
    }

    void* allocate(size_t size) const
    {
        void* result = std::malloc(size);

        WHIRL_TRACE("alloc host [pointer: host_{0:p}, size: {1}] {2}", result, size, (result ? "success" : " FAILURE"));

        return result;
    }

    void deallocate(void* ptr) const
    {
        WHIRL_TRACE("dealloc host [pointer: host_{0:p}]", ptr);
        std::free(ptr);
    }
    
    void memory_set(void* dst, int value, size_t size, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE("memset host [pointer: host_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        std::memset(dst, value, size);
    }
   
    void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE    ("memcpy from device to host [destination: host_{0:p}, source: device_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
    
    void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE    ("memcpy from host to device [destination: device_{0:p}, source: host_{1:p}, size: {2}]", dst, src, size);
        WHIRL_CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
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

    static bool is_supported(int device) 
    { 
        return true; 
    }

    void* allocate(size_t size, unsigned int flags = cudaHostAllocDefault) const
    {
        void* result = nullptr;

        WHIRL_CUDA_CALL(cudaHostAlloc(&result, size, flags));
        WHIRL_TRACE    ("alloc pinned [pointer: pinned_{0:p}, size: {1}, flags: {2:d}] {3}", result, size, flags, (result ? "success" : " FAILURE"));

        return result;
    }

    void deallocate(void* ptr) const
    {
        WHIRL_TRACE    ("dealloc pinned [pointer: pinned_{0:p}]", ptr);
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    void memory_set(void* dst, int value, size_t size, cudaStream_t stream) const
    {
        WHIRL_TRACE    ("memset pinned [pointer: pinned_{0:p}, size: {1}, value: {2:#x}, stream_handle: {3:#x}]", dst, size, value, stream);
        WHIRL_CUDA_CALL(cudaMemsetAsync(dst, value, size, stream));
    }

    void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream) const
    {
        WHIRL_TRACE("memcpy from device to pinned [destination: pinned_{0:p}, source: device_{1:p}, "
                                                  "size: {2}, stream_handle: {3:#x}]", dst, src, size, stream);
        WHIRL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    }
    
    void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) const
    {
        WHIRL_TRACE("memcpy from pinned to device [destination: device_{0:p}, source: pinned_{1:p}, "
                                                   "size: {2}, stream_handle: {3:#x}]", dst, src, size, stream);
        WHIRL_CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    }
};

template<>
class WHHostMemoryManager<WHAllocType::MANAGED> : public WHSingletonMemoryManager<WHHostMemoryManager<WHAllocType::MANAGED>>
{
private:
    friend class WHSingletonMemoryManager<WHHostMemoryManager>;

    WHHostMemoryManager(): WHSingletonMemoryManager(WHAllocType::MANAGED) {}

public:
    virtual ~WHHostMemoryManager() = default;
    
    static bool is_supported(int device) 
    {
        int result = 0;

        cudaDeviceGetAttribute(&result, cudaDevAttrManagedMemory, device);

        return !!result; 
    }

    void* allocate(size_t size, unsigned int flags = cudaMemAttachGlobal) const
    {
        void* result = nullptr;

        WHIRL_CUDA_CALL(cudaMallocManaged(&result, size, flags));
        WHIRL_TRACE    ("alloc managed [pointer: device_{0:p}, size: {1}, flags: {2:d}] {3}", result, size, flags, (result ? "success" : " FAILURE"));

        return result;
    }

    void deallocate(void* ptr) const
    {
        WHIRL_TRACE    ("dealloc managed [pointer: managed_{0:p}]", ptr);
        WHIRL_CUDA_CALL(cudaFree(ptr));
    }
    
    void memory_set(void* dst, int value, size_t size) const
    {
        WHIRL_TRACE    ("memset managed [pointer: managed_{0:p}, size: {1}, value: {2:#x}]", dst, size, value);
        WHIRL_CUDA_CALL(cudaMemset(dst, value, size));
    }

    void prefetch_to_host(const void* src, size_t size, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE    ("prefetch to host [source: managed_{0:p}, size: {1}, device: cudaCpuDeviceId, stream: {2:#x}]", src, size, stream);
        WHIRL_CUDA_CALL(cudaMemPrefetchAsync(src, size, cudaCpuDeviceId, stream));
    }
    
    void prefetch_to_device(const void* src, size_t size, int device, cudaStream_t stream = nullptr) const
    {
        WHIRL_TRACE    ("prefetch to device [source: managed_{0:p}, size: {1}, device: {2:d}, stream: {3:#x}]", src, size, device, stream);
        WHIRL_CUDA_CALL(cudaMemPrefetchAsync(src, size, device, stream));
    }

    void mem_advice(const void* src, size_t size, cudaMemoryAdvise advice, int device = cudaCpuDeviceId)
    {
        WHIRL_TRACE    ("memory advice [source: managed_{0:p}, size: {1}, advice: {2:d}, device: {3:d}]", src, size, advice, device);
        WHIRL_CUDA_CALL(cudaMemAdvise(src, size, advice, device));
    }
};

class WHDeviceMemoryManager : public WHSingletonMemoryManager<WHDeviceMemoryManager>
{
private:
    friend class WHSingletonMemoryManager<WHDeviceMemoryManager>;

    WHDeviceMemoryManager(): WHSingletonMemoryManager(WHAllocType::DEVICE) {}

public:
    virtual ~WHDeviceMemoryManager() = default;
    
    static bool is_supported(int device) 
    {
        return true;
    }

    void* allocate(size_t size) const
    {
        void* result = nullptr;

        WHIRL_CUDA_CALL(cudaMalloc(&result, size));
        WHIRL_TRACE    ("alloc device [pointer: device_{0:p}, size: {1}] {2}", result, size, (result ? "success" : " FAILURE"));

        return result;
    }

    void deallocate(void* ptr) const
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