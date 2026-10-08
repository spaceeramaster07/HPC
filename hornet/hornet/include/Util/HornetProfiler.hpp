#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace hornet {

struct PhaseMeasurement {
    std::string phase_name;
    double duration_ms;
};

class HornetProfiler {
private:
    bool _enabled = false;
    bool _nvtx_enabled = true;
    std::unordered_map<std::string, double> _phase_times_ms;
    std::vector<PhaseMeasurement> _history;

    HornetProfiler() = default;

public:
    static HornetProfiler& instance() {
        static HornetProfiler inst;
        return inst;
    }

    void set_enabled(bool val) { _enabled = val; }
    bool is_enabled() const { return _enabled; }

    void set_nvtx_enabled(bool val) { _nvtx_enabled = val; }
    bool is_nvtx_enabled() const { return _nvtx_enabled; }

    void clear() {
        _phase_times_ms.clear();
        _history.clear();
    }

    void record_phase(const std::string& name, double duration_ms) {
        _phase_times_ms[name] = duration_ms;
        _history.push_back({name, duration_ms});
    }

    double get_phase_ms(const std::string& name) const {
        auto it = _phase_times_ms.find(name);
        if (it != _phase_times_ms.end()) {
            return it->second;
        }
        return 0.0;
    }

    const std::unordered_map<std::string, double>& get_all_phases() const {
        return _phase_times_ms;
    }

    const std::vector<PhaseMeasurement>& get_history() const {
        return _history;
    }
};

class ScopedPhaseTimer {
private:
    std::string _name;
    bool _is_gpu;
    bool _active;
    std::chrono::high_resolution_clock::time_point _start_time;

public:
    ScopedPhaseTimer(const std::string& name, bool is_gpu = false)
        : _name(name), _is_gpu(is_gpu), _active(HornetProfiler::instance().is_enabled()) {
        if (_active) {
            if (HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePushA(_name.c_str());
            }
            if (_is_gpu) {
                cudaDeviceSynchronize();
            }
            _start_time = std::chrono::high_resolution_clock::now();
        }
    }

    ~ScopedPhaseTimer() {
        if (_active) {
            if (_is_gpu) {
                cudaDeviceSynchronize();
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(end_time - _start_time).count();
            if (HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePop();
            }
            HornetProfiler::instance().record_phase(_name, duration_ms);
        }
    }
};

} // namespace hornet
