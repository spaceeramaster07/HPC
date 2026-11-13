cmake_minimum_required(VERSION 3.18)
project(HornetBaseline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(baseline_benchmark ../algorithms/benchmark.cu)

target_include_directories(baseline_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../algorithms
)

target_compile_definitions(baseline_benchmark PRIVATE
    HORNET_DS_NS=baseline
)

target_compile_options(baseline_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)

target_link_libraries(baseline_benchmark PRIVATE
    CUDA::cudart
    CUDA::nvtx3
    ${CMAKE_DL_LIBS}
)
