#pragma once

#include <memory>
#include <functional>

#include "WHAbstractMemoryStrategy.h"
#include "WHAbstractScene.h"
#include "memory/WHEvent.h"

namespace whirl {

template<class SceneType, class MemoryStrategyType>
class WHEngine
{
public:
    typedef BufferCallback_t std::function<void(const WHBuffer&)>;

    WHEngine() = default;
    WHEngine(std::unique_ptr<SceneType>&& scene_set, std::unique_ptr<MemoryStrategyType>&& memory_strategy_set, 
             const BufferCallback_t& callback_set = {}): 
        scene_(scene_set), memory_strategy_(memory_strategy_set), synchronizer_(memory_strategy_->get_synchronizer()), callback_(callback_set) {}
    
    virtual ~WHEngine() = default;

    void render()
    {
        memory_strategy_->process_buffer(scene_.get()); synchronizer_->synchronize();
        callback_(memory_strategy_->get_buffer());
    }
    
    void set_callback(const BufferCallback_t& callback_set)
    {
        callback_ = callback_set;
    }

    void set_synchronizer(const std::shared_ptr<WHEvent> synchronizer_set)
    {
        synchronizer_ = synchronizer_set;
        memory_strategy_->set_synchronizer(synchronizer_);
    }

    template<typename... Types>
    void set_memory_strategy(Types&&... args) 
    { 
        memory_strategy_ = std::make_unique<MemoryStrategyType<SceneType>>(std::forward<Types>(args)...);
        memory_strategy_->set_synchronizer();
    }
    
          SceneType* get_scene_ptr()       { return scene_.get(); }
    const SceneType* get_scene_ptr() const { return scene_.get(); }
    
    const std::unique_ptr<MemoryStrategyType>& get_memory_strategy() const { return memory_strategy_; }
    
private:
    std::unique_ptr<SceneType>          scene_;
    std::unique_ptr<MemoryStrategyType> memory_strategy_;

    std::shared_ptr<WHEvent> synchronizer_;
    BufferCallback_t         callback_;
};

}//namespace whirl