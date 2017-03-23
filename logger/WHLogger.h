#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

#define SPDLOG_DEBUG_ON
#define SPDLOG_TRACE_ON

#include <spdlog\spdlog.h>

#define WHIRL_TRACE_ON
#define WHIRL_DEBUG_ON

//TODO enable trace and debug in spdlog
class WHGlobalLogger
{
public:
    static WHGlobalLogger& instance()
    {
        static WHGlobalLogger instance_ = WHGlobalLogger("logs/logfile");
        
        return instance_;
    }
    
    template<typename ...Types>
    void debug(const char* format, Types&& ...args)
    {
        spdlog_logger_->debug(format, std::forward<args>...);
    }

    template<typename ...Types>
    void trace(const char* format, Types&& ...args)
    {
        spdlog_logger_->trace(format, std::forward<args>...);
    }

    cudaError_t debug_cuda_call(const char* prefix_str, cudaError_t call_result)
    {
        spdlog_logger_->debug("CUDA_CALL {} {}", prefix_str, (call_result == cudaSuccess ? "OK" : "ERROR"));

        return call_result;
    }
    
    cudaError_t trace_cuda_call(const char* prefix_str, cudaError_t call_result)
    {
        spdlog_logger_->trace("CUDA_CALL {} {}", prefix_str, cudaGetErrorString(call_result));

        return call_result;
    }

private:
    WHGlobalLogger(std::string log_file_name_set):
        spdlog_logger_(spdlog::basic_logger_mt("WHGlobalLogger", log_file_name_set))
    {
#if defined WHIRL_TRACE_ON
        spdlog_logger_->set_level(spdlog::level::trace);
#elif defined WHIRL_DEBUG_ON
        spdlog_logger_->set_level(spdlog::level::debug);
#else
        spdlog_logger_->set_level(spdlog::level::info);
#endif

        spdlog_logger_->debug("{:-^128}",  "start WHGlobalLogger logging");
    }
    
    ~WHGlobalLogger()
    {
        spdlog_logger_->debug("{:-^128}\n", "end WHGlobalLogger logging");
    }

    std::shared_ptr<spdlog::logger> spdlog_logger_;
};

#if defined WHIRL_TRACE_ON
    #define WHIRL_TRACE(format, ...) \
        (WHGlobalLogger::instance().trace(" [func " __FUNCTION__ ", file " __FILE__ ", line " _STRINGIZE(__LINE__) "] " format, __VA__ARGS__))
#else
    #define WHIRL_TRACE(format, ...)
#endif

#if defined WHIRL_DEBUG_ON
    #define WHIRL_DEBUG(format, ...) \
        (WHGlobalLogger::instance().debug(" [func " __FUNCTION__ ", file " __FILE__ ", line " _STRINGIZE(__LINE__) "] " format, __VA__ARGS__))
#else
    #define WHIRL_DEBUG(format, ...)
#endif

#if defined WHIRL_TRACE_ON
    #define WHIRL_CUDA_CALL(cuda_call) \
        (WHGlobalLogger::instance().trace_cuda_call(" [func " __FUNCTION__ ", file " __FILE__ \
                                                    ", line " _STRINGIZE(__LINE__) ", call " _STRINGIZE(cuda_call) "] ", cuda_call))
#elif defined WHIRL_DEBUG_ON
    #define WHIRL_CUDA_CALL(cuda_call) \
        (WHGlobalLogger::instance().debug_cuda_call(" [func " __FUNCTION__ ", file " __FILE__ \
                                                    ", line " _STRINGIZE(__LINE__) ", call " _STRINGIZE(cuda_call) "] ", cuda_call))
#else
    #define WHIRL_CUDA_CALL(cuda_call) cuda_call
#endif
