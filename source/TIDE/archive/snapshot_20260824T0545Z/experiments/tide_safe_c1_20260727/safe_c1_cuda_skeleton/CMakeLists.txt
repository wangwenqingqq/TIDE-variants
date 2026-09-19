cmake_minimum_required(VERSION 3.24)

project(GTS_safe_c1_trace LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "Target CUDA architecture")

set(GTS_SOURCE_ROOT
    "/workspace/experiments/tide_safe_c1_20260727/source_gts_incremental"
    CACHE PATH "Read-only GTS source archive used for static-tree headers")

if(NOT EXISTS "${GTS_SOURCE_ROOT}/include/tree.cuh")
  message(FATAL_ERROR "GTS_SOURCE_ROOT must point at an isolated GTS source tree")
endif()

add_executable(GTS_safe_c1_trace src/safe_c1_trace.cu)
target_include_directories(GTS_safe_c1_trace PRIVATE "${GTS_SOURCE_ROOT}/include")
target_compile_options(GTS_safe_c1_trace PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
set_target_properties(GTS_safe_c1_trace PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
