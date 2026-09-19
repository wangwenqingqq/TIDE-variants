cmake_minimum_required(VERSION 3.27)
project(RTTIDEFusedProbe LANGUAGES CUDA CXX)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
add_executable(rt_tide_fused_probe rt_tide_fused_probe.cu)
target_include_directories(rt_tide_fused_probe PRIVATE
  ${RTSPATIAL_ROOT}/include
  ${OPTIX_ROOT}/include)
target_compile_definitions(rt_tide_fused_probe PRIVATE PTX_ROOT="${PTX_ROOT}")
target_compile_options(rt_tide_fused_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(rt_tide_fused_probe PROPERTIES
  CUDA_ARCHITECTURES 120
  CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(rt_tide_fused_probe PRIVATE CUDA::cuda_driver CUDA::cudart)
