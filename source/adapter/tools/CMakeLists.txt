cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(gpusim_direct_exact LANGUAGES CXX CUDA)

if(NOT DEFINED GPUSIMILARITY_SOURCE)
  message(FATAL_ERROR "Set GPUSIMILARITY_SOURCE to the pinned upstream checkout")
endif()

find_package(Qt5Core REQUIRED)
find_package(Qt5Concurrent REQUIRED)

add_executable(gpusim_direct_exact
  direct_exact_benchmark.cpp
  "${GPUSIMILARITY_SOURCE}/fingerprintdb_cuda.cpp"
  "${GPUSIMILARITY_SOURCE}/fingerprintdb_cuda.cu"
  "${GPUSIMILARITY_SOURCE}/calculation_functors.cpp"
)

target_include_directories(gpusim_direct_exact PRIVATE
  "${GPUSIMILARITY_SOURCE}"
)
target_link_libraries(gpusim_direct_exact PRIVATE Qt5::Core Qt5::Concurrent)
target_compile_features(gpusim_direct_exact PRIVATE cxx_std_17)
set_target_properties(gpusim_direct_exact PROPERTIES
  # CUDA 13.1's bundled CCCL/Thrust requires C++17. This changes only the
  # compilation dialect; the pinned upstream scoring/result sources remain
  # byte-for-byte unmodified.
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON
)

# The released core uses only ordinary CUDA instructions; no architecture-
# accelerated feature is required. CUDA 13.1 accepts compute capability 12.0
# for the frozen RTX PRO 6000 Blackwell host.
set_property(TARGET gpusim_direct_exact PROPERTY CUDA_ARCHITECTURES 120)

target_compile_options(gpusim_direct_exact PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
  # The 2024 source relied on a transitive Thrust include that CUDA 13.1 no
  # longer provides. Pre-including the public header is a build-compatibility
  # shim; the pinned source files remain unmodified.
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--pre-include=thrust/remove.h>
)
