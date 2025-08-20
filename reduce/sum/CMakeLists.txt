cmake_minimum_required(VERSION 3.16)
project(SUM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

add_executable(sum sum.cu ${SRC})
