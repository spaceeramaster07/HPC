#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace hornet_common {

using vid_t = int;
using eoff_t = int;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%s)\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(err__), #call); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

inline int ceil_log2(int x) {
    if (x <= 1) return 0;
    return 32 - __builtin_clz(static_cast<unsigned int>(x - 1));
}

inline int bsize_for_deg(int deg) {
    return 1 << ceil_log2(deg);
}

struct PhaseMeasurement {
    std::string phase_name;
    double duration_ms;
};

class HornetProfiler {
public:
    static HornetProfiler& instance() {
        static HornetProfiler inst;
        return inst;
    }

    void set_enabled(bool val) { enabled_ = val; }
    bool is_enabled() const { return enabled_; }

    void set_nvtx_enabled(bool val) { nvtx_enabled_ = val; }
    bool is_nvtx_enabled() const { return nvtx_enabled_; }

    void clear() {
        phase_times_ms_.clear();
        history_.clear();
    }

    void record_phase(const std::string& name, double duration_ms) {
        phase_times_ms_[name] = duration_ms;
        history_.push_back({name, duration_ms});
    }

    double get_phase_ms(const std::string& name) const {
        auto it = phase_times_ms_.find(name);
        return (it != phase_times_ms_.end()) ? it->second : 0.0;
    }

    const std::unordered_map<std::string, double>& get_all_phases() const {
        return phase_times_ms_;
    }

    const std::vector<PhaseMeasurement>& get_history() const {
        return history_;
    }

private:
    HornetProfiler() = default;
    bool enabled_ = false;
    bool nvtx_enabled_ = true;
    std::unordered_map<std::string, double> phase_times_ms_;
    std::vector<PhaseMeasurement> history_;
};

class ScopedPhaseTimer {
public:
    ScopedPhaseTimer(const std::string& name, bool is_gpu = false)
        : name_(name), is_gpu_(is_gpu), active_(HornetProfiler::instance().is_enabled()) {
        if (active_) {
            if (HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePushA(name_.c_str());
            }
            if (is_gpu_) {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }

    ~ScopedPhaseTimer() {
        if (active_) {
            if (is_gpu_) {
                cudaDeviceSynchronize();
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration_ms =
                std::chrono::duration<double, std::milli>(end_time - start_time_).count();
            if (HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePop();
            }
            HornetProfiler::instance().record_phase(name_, duration_ms);
        }
    }

private:
    std::string name_;
    bool is_gpu_;
    bool active_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace hornet_common
