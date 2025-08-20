cmake_minimum_required(VERSION 3.16)
project(CUDA_SGEMM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(cublas_exmple cublas_exmple.cu)

# link cublas library
target_link_libraries(cublas_exmple ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})