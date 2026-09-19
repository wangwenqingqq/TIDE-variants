cmake_minimum_required(VERSION 3.24)
project(p1_bridges LANGUAGES CXX CUDA)
if(NOT DEFINED GPUSIMILARITY_SOURCE)
  message(FATAL_ERROR "Pinned GPUSIMILARITY_SOURCE is required")
endif()
find_package(Qt5Core REQUIRED)
find_package(Qt5Concurrent REQUIRED)
add_library(p1_gpusim SHARED gpusim_bridge.cpp
  "${GPUSIMILARITY_SOURCE}/fingerprintdb_cuda.cpp"
  "${GPUSIMILARITY_SOURCE}/fingerprintdb_cuda.cu"
  "${GPUSIMILARITY_SOURCE}/calculation_functors.cpp")
target_include_directories(p1_gpusim PRIVATE "${GPUSIMILARITY_SOURCE}")
target_link_libraries(p1_gpusim PRIVATE Qt5::Core Qt5::Concurrent)
set_target_properties(p1_gpusim PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17
  CXX_STANDARD_REQUIRED YES CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES 120
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(p1_gpusim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--pre-include=thrust/remove.h>)
add_library(p1_tide SHARED tide_bridge.cu)
set_target_properties(p1_tide PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
  CUDA_ARCHITECTURES 120 POSITION_INDEPENDENT_CODE ON)
target_compile_options(p1_tide PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--ptxas-options=-v>)
