cmake_minimum_required(VERSION 3.18)
project(HornetImproved LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(improved_benchmark ../algorithms/benchmark.cu)

target_include_directories(improved_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../algorithms
)

target_compile_definitions(improved_benchmark PRIVATE
    HORNET_DS_NS=improved
)

target_compile_options(improved_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_libraries(improved_benchmark PRIVATE
    CUDA::cudart
    CUDA::nvtx3
    OpenMP::OpenMP_CXX
    ${CMAKE_DL_LIBS}
)
