#pragma once

#include <memory>

#include "memory/WHEvent.h"
#include "memory/WHMemoryManager.h"
#include "render/WHRay.h"
#include "render/WHCamera.h"
#include "render/WHGraphicObject.h"
#include "render/WHLight.h"
#include "render/math/WHVector.h"

namespace whirl {

template<class... ObjectTypes> struct WHObjectBuffer;
template<class... ObjectTypes> class WHVolume;

class WHLightStrategy;

template<class VolumeType, class LightStrategyType> class WHScene;

template<class... ObjectTypes>
__device__ void wh_device_get_light_properties(const typename WHObjectBuffer<ObjectTypes...>::NearestObjectData* nearest_object_data_ptr, 
                                               const WHVector3& position, WHNormal3* normal_ptr_ret, WHMaterial* material_ptr_ret, float min_distance/* = FLT_MAX*/);

template<class... ObjectTypes>
__device__ float wh_device_distance_function(const WHObjectBuffer<ObjectTypes...>* object_buffer_ptr, const WHVector3& position, 
                                             typename WHObjectBuffer<ObjectTypes...>::NearestObjectData* nearest_object_data_ptr_ret);

template<class... ObjectTypes>
__device__ float wh_device_step(const WHVolume<ObjectTypes...>::SGpuData* volume_ptr, WHRay* ray_ptr, WHNormal3* normal_ptr_ret, WHMaterial* material_ptr_ret);

__device__ WHColor wh_device_get_color(const WHLightStrategy::SGpuData* light_strategy_ptr, const WHMaterial& material, 
                                       const WHNormal3& normal, const WHRay& ray);

template<class VolumeType, class LightStrategyType>
__device__ WHColor wh_device_trace_ray(const VolumeType::SGpuData* volume_ptr, const LightStrategyType::SGpuData* light_strategy_ptr, WHRay* ray_ptr);

template<class... ObjectTypes>
struct WHObjectBuffer: public WHObjectBuffer<ObjectTypes>...
{
    struct NearestObjectData : public WHObjectBuffer<ObjectTypes>::NearestObjectData...
    {
        float min_distance;
    };

    template<class ExtractObjectType> 
    __host__ __device__ size_t get_buffer_size() const { return WHObjectBuffer<ExtractObjectType>::buffer_size; }
    
    template<class ExtractObjectType> __host__ __device__       ExtractObjectType* get_buffer()       { return WHObjectBuffer<ExtractObjectType>::buffer; }
    template<class ExtractObjectType> __host__ __device__ const ExtractObjectType* get_buffer() const { return WHObjectBuffer<ExtractObjectType>::buffer; }
};

template<class ObjectType> 
struct WHObjectBuffer<ObjectType>
{
    struct NearestObjectData
    {
        ObjectType* object_ptr; 
        float distance;
    };
    
    template<class ExtractObjectType>//ExtractObjectType must be same to ObjectType 
    __host__ __device__ size_t get_buffer_size() const { return WHObjectBuffer<ExtractObjectType>::buffer_size; }
    
    template<class ExtractObjectType> __host__ __device__       ExtractObjectType* get_buffer()       { return WHObjectBuffer<ExtractObjectType>::buffer; }
    template<class ExtractObjectType> __host__ __device__ const ExtractObjectType* get_buffer() const { return WHObjectBuffer<ExtractObjectType>::buffer; }


    size_t      buffer_size;
    ObjectType* buffer;
};

template<class... ObjectTypes>
__device__ void wh_device_get_light_properties(const typename WHObjectBuffer<ObjectTypes...>::NearestObjectData* nearest_object_data_ptr, 
                                               const WHVector3& position, WHNormal3* normal_ptr_ret, WHMaterial* material_ptr_ret, float min_distance = FLT_MAX)
{
    float min_distance = nearest_object_data_ptr->min_distance;

    wh_device_get_light_properties<ObjectTypes>(dynamic_cast<const typename WHObjectBuffer<ObjectTypes>::NearestObjectData*>(nearest_object_data_ptr),
                                                position, normal_ptr_ret, material_ptr_ret, min_distance)...;
}

template<class ObjectType>
__device__ void wh_device_get_light_properties<ObjectType>(typename const WHObjectBuffer<ObjectType>::NearestObjectData* nearest_object_data_ptr, 
                                                           const WHVector3& position, WHNormal3* normal_ptr_ret, WHMaterial* material_ptr_ret, float min_distance = FLT_MAX)
{
    if (nearest_object_data_ptr->distance > min_distance) 
        return;

    *normal_ptr_ret   = nearest_object_data_ptr->object_ptr->get_normal(position);
    *material_ptr_ret = nearest_object_data_ptr->object_ptr->get_material();
}

template<class... ObjectTypes>
class WHVolume
{
public:
    struct SGpuData
    {
        WHObjectBuffer<ObjectTypes...> object_buffer_;

        WHVector3 bounds_;
        size_t max_iter_;
    };

public:
    WHVolume() = default;
    WHVolume(size_t object_buffer_sizes_set[sizeof...(ObjectTypes)], WHVector3 bounds_set, size_t max_iter_set = 64u): 
        memory_manager_(std::dynamic_pointer_cast<WHAbstractMemoryManager>(WHMemoryManager<WHAllocType::PINNED>::instance())), 
        gpu_data_ptr_  {static_cast<SGpuData*>(memory_manager_->allocate(sizeof(SGpuData), cudaHostAllocMapped))} 
    {
        size_t index = 0u;
        gpu_data_ptr_->object_buffer_.WHObjectBuffer<ObjectTypes>>::buffer_size = object_buffer_sizes_set[index++]...;
        gpu_data_ptr_->object_buffer_.WHObjectBuffer<ObjectTypes>>::buffer = static_cast<ObjectTypes*>
            (memory_manager_->allocate(gpu_data_ptr_->object_buffer_.WHObjectBuffer<ObjectTypes>>::buffer_size * sizeof(ObjectTypes), 
             cudaHostAllocMapped))...; 
                                           
        gpu_data_ptr_->bounds_   = bounds_set;
        gpu_data_ptr_->max_iter_ = max_iter_set; 
    }

    virtual ~WHVolume()
    {
        if (gpu_data_ptr)
        {
            memory_manager_->deallocate(gpu_data_ptr_->object_buffer_.WHObjectBuffer<ObjectTypes>::buffer)...;

            *gpu_data_ptr_ = {};
        
            memory_manager_->deallocate(gpu_data_ptr_);
        }

        gpu_data_ptr = nullptr;
    }

    std::shared_ptr<WHAbstractMemoryManager> get_memory_manager() const { return memory_manager_; }

    const SGpuData* get_gpu_data_ptr() const { return get_gpu_data_ptr_; }
          SGpuData* get_gpu_data_ptr()       { return get_gpu_data_ptr_; }
    
private:
    std::shared_ptr<WHAbstractMemoryManager> memory_manager_;

    SGpuData* gpu_data_ptr_;
};

template<class... ObjectTypes>
__device__ float wh_device_distance_function(const WHObjectBuffer<ObjectTypes...>* object_buffer_ptr, const WHVector3& position, 
                                             typename WHObjectBuffer<ObjectTypes...>::NearestObjectData* nearest_object_data_ptr_ret)
{
    float result = FLT_MAX;
    float dist_array[sizeof...(ObjectTypes)] = 
        { wh_device_distance_function<ObjectTypes>(dynamic_cast<const WHObjectBuffer<ObjectTypes>*>(object_buffer_ptr), position, 
                                                   dynamic_cast<typename WHObjectBuffer<ObjectTypes>::NearestObjectData*>(nearest_object_data_ptr_ret))... };

    for (size_t i = 0u; i < sizeof...(ObjectTypes); i++)
    {
        if (result > dist_array[i])
            result = dist_array[i];
    }
    
    nearest_object_data_ptr_ret->min_distance = result;

    return result;
}

template<class ObjectType>
__device__ float wh_device_distance_function<ObjectType>(const WHObjectBuffer<ObjectType>* object_buffer_ptr, const WHVector3& position, 
                                                         typename WHObjectBuffer<ObjectType>::NearestObjectData* nearest_object_data_ptr_ret)
{
    float result = FLT_MAX, cur_dist = FLT_MAX;
    
    for (size_t i = 0u; i < object_buffer_ptr->object_buffer_size_; i++)
    {
        cur_dist = object_buffer_ptr->object_buffer_[i].device_distance_function(position);
            
        if (cur_dist < result)
        {
            result = cur_dist;
            nearest_object_data_ptr_ret->object_ptr = &(object_buffer_ptr->object_buffer_[i]);
        }
    }

    nearest_object_data_ptr_ret->distance = result;
    
    return result;
}

template<class... ObjectTypes>
__device__ float wh_device_step(const WHVolume<ObjectTypes...>::SGpuData* volume_ptr, WHRay* ray_ptr, WHNormal3* normal_ptr_ret, WHMaterial* material_ptr_ret)
{
    typename WHObjectBuffer<ObjectTypes...>::NearestObjectData nearest_object_data;

    float result = 0.0f, incr = fabs(wh_device_distance_function(volume_ptr->object_buffer_, ray_ptr->position, &nearest_object_data));
        
    for (size_t i = 0u; incr > result*FLT_EPSILON && i < volume_ptr->max_iter_; i++)
    {
        result += incr;
        ray_ptr->step(incr);
        incr = wh_device_distance_function(volume_ptr->object_buffer_, ray_ptr->position, &nearest_object_data);

        if (!is_inside(ray_ptr->position, volume_ptr->bounds_)) return 0.0f;
    }
    
    wh_device_get_light_properties(&nearest_object_data, ray_ptr->position, normal_ptr_ret, material_ptr_ret);
    
    return result;
}
    
class WHLightStrategy
{
public:
    struct SGpuData
    {
        size_t   lighters_buffer_size_;
        WHLight* lighters_buffer_;
    };
    
public:
    WHLightStrategy() = default;
    WHLightStrategy(size_t lighters_buffer_size_set):
        memory_manager_(std::dynamic_pointer_cast<WHAbstractMemoryManager>(WHMemoryManager<WHAllocType::PINNED>::instance())), 
        gpu_data_ptr_  {static_cast<SGpuData*>(memory_manager_->allocate(sizeof(SGpuData), cudaHostAllocMapped))} 
    {
        gpu_data_ptr_->lighters_buffer_size_ = lighters_buffer_size_set; 
        gpu_data_ptr_->lighters_buffer_      = static_cast<WHLight*>(memory_manager_->allocate(gpu_data_ptr_->lighters_buffer_size_ * sizeof(WHLight), 
                                                                     cudaHostAllocMapped));
    }

    ~WHLightStrategy()
    {
        if (gpu_data_ptr_)
        {   
            if (gpu_data_ptr_->lighters_buffer_)
                memory_manager_->deallocate(gpu_data_ptr_->lighters_buffer_);

            gpu_data_ptr_->lighters_buffer_      = nullptr;
            gpu_data_ptr_->lighters_buffer_size_ = {};

            memory_manager_->deallocate(gpu_data_ptr_);
        }

        gpu_data_ptr_ = nullptr;
    }

    const SGpuData* get_gpu_data_ptr() const { return gpu_data_ptr_; }
          SGpuData* get_gpu_data_ptr()       { return gpu_data_ptr_; }

private:
    std::shared_ptr<WHAbstractMemoryManager> memory_manager_;//TODO: create memory strategies for mapped and managed alloc types

    SGpuData* gpu_data_ptr_;
};

__device__ WHColor wh_device_get_color(const WHLightStrategy::SGpuData* light_strategy_ptr, const WHMaterial& material, 
                                       const WHNormal3& normal, const WHRay& ray)
{
    WHColor result = WH_BLACK;

    for (size_t i = 0u; i < light_strategy_ptr->lighters_buffer_size_; i++)
        result += light_strategy_ptr->lighters_buffer_[i].get_point_color(material, normal, ray);

    result.w = 1.0f;
        
    truncate(result);

    return result;
}

template<class VolumeType, class LightStrategyType>
class WHScene : private VolumeType, private LightStrategyType
{
public:
    WHScene() = default;
    WHScene(const VolumeType& volume_set, const LightStrategyType& light_strategy_set, const std::shared_ptr<WHEvent>& synchronizer_set, 
            const std::shared_ptr<WHAbstractMemoryManager>& memory_manager_set, unsigned int alloc_flags, 
            const std::shared_ptr<WHCamera>& camera_set):
        VolumeType(volume_set), LightStrategyType(light_strategy_set),
        synchronizer_(synchronizer_set), memory_manager_(memory_manager_set), 
        camera_(camera_set) {}

    virtual ~WHScene() = default;
        
    size_t get_light_source_buffer_size() const { return light_source_buffer_size_; }
    
    const WHGraphicObject* get_light_source_buffer_() const { return light_source_buffer_; }
          WHGraphicObject* get_light_source_buffer_()       { return light_source_buffer_; }
    
protected:
    std::shared_ptr<WHEvent>                 synchronizer_;
    std::shared_ptr<WHAbstractMemoryManager> memory_manager_;
    
    std::shared_ptr<WHCamera> camera_;
};

template<class VolumeType, class LightStrategyType>
__device__ WHColor wh_device_trace_ray(const VolumeType::SGpuData* volume_ptr, const LightStrategyType::SGpuData* light_strategy_ptr, WHRay* ray_ptr)
{
    WHColor result_color = WH_BLACK;
    WHGraphicObject* collision_object_ptr = nullptr;
    float cur_factor = 1.0f;

    WHNormal3  cur_normal;
    WHMaterial cur_material;

    while (is_inside(ray_ptr->position, VolumeType::get_bounds()))
    {
        float step_len = wh_device_step(volume_ptr, &ray_ptr, &cur_normal, &cur_material);

        if (step_len == 0.0f) break;
        if (cur_material.mat_transparency < 1.0f - cur_color.w) break;

        result_color += wh_device_get_color(light_strategy_ptr, cur_material, cur_normal, *ray_ptr) * cur_factor;//must return normalized
            
        cur_factor *= collision_object_ptr->cur_material.mat_transparency;

        ray_ptr->refract(cur_material, cur_normal);
        ray_ptr->step(4*step_len*FLT_EPSILON);
    }

    normalize(result_color);

    return result_color;
}


}//namespace whirl