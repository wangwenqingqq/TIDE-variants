cmake_minimum_required(VERSION 3.20)

# Isolated C1 microbenchmark worktree; the archive source remains untouched.
set(CMAKE_CUDA_ARCHITECTURES 120)
project(GTS LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION ON)

option(C1_PERSISTENT_WORKSPACE "Enable C1 capacity-persistent query workspace" ON)
option(C1_ONE_QUERY_FASTPATH "Enable C1 qnum==1 fast path" ON)

if(C1_PERSISTENT_WORKSPACE)
  set(C1_PERSISTENT_WORKSPACE_VALUE 1)
else()
  set(C1_PERSISTENT_WORKSPACE_VALUE 0)
endif()
if(C1_ONE_QUERY_FASTPATH)
  set(C1_ONE_QUERY_FASTPATH_VALUE 1)
else()
  set(C1_ONE_QUERY_FASTPATH_VALUE 0)
endif()

include_directories(include)
add_executable(GTS src/main.cu)
add_executable(C1Microbench src/c1_microbench.cu)

# C1Microbench output is build-directory-local, avoiding cross-variant overwrite.
set_target_properties(C1Microbench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(C1Microbench PRIVATE
  C1_PERSISTENT_WORKSPACE=${C1_PERSISTENT_WORKSPACE_VALUE}
  C1_ONE_QUERY_FASTPATH=${C1_ONE_QUERY_FASTPATH_VALUE})
foreach(target GTS C1Microbench)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
endforeach()
