cmake_minimum_required(VERSION 3.2)

# 设置CUDA架构，针对NVIDIA RTX PRO 6000 Black (计算能力8.6)
# 必须在project()命令之前设置
set(CMAKE_CUDA_ARCHITECTURES 120)

project(GTS LANGUAGES CXX CUDA)

# 原来是 14 和 11，改为 17 以适配 CUDA 13.1
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17) 
set(CUDA_SEPARABLE_COMPILATION ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# The legacy GTS executable is header-defined around one entry point. Do not
# auto-glob src/: the standalone C1 harness is an independent executable.
set(GTS_SRC src/main.cu)
include_directories(include)

#find_package(OpenCV 4 REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#message(STATUS "include:${OpenCV_INCLUDE_DIRS}")
#link_libraries(${OpenCV_LIBS})

add_executable(${PROJECT_NAME} ${GTS_SRC})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
# Minimal source-bound CUDA closure for C1 lifecycle semantics.  This target is
# deliberately separate from GTS until native traversal is wired to the same
# Base+Delta+overlay merge contract.
add_executable(C1MinimalClosure src/c1_minimal_closure.cu)
target_compile_options(C1MinimalClosure PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
