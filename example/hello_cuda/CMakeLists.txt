cmake_minimum_required(VERSION 3.16)
project(HELLO_CUDA LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(hello_cuda hello_cuda.cu)
