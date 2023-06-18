set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(body_lib
        VERSION 1.0.0
        DESCRIPTION "Body library"
        LANGUAGES CUDA)

add_library(body STATIC Body.cu)

target_link_libraries(body PUBLIC material geometry)

find_package(Eigen3 REQUIRED)
target_link_libraries(body PUBLIC Eigen3::Eigen)

set_target_properties(body
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(body
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(body
        PROPERTIES
        LINKER_LANGUAGE CUDA)