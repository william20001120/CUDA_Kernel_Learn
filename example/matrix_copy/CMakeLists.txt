cmake_minimum_required(VERSION 3.16)
project(MATRIX_COPY LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(matrix_copy matrix_copy.cu)
