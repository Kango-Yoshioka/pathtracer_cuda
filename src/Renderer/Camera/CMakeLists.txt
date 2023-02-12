set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(camera_lib
        VERSION 1.0.0
        DESCRIPTION "Object library"
        LANGUAGES CUDA)

add_library(camera STATIC Camera.cu)

target_include_directories(camera PUBLIC ../../../third_party/eigen-3.4.0)


set_target_properties(camera
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(camera
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(camera
        PROPERTIES
        LINKER_LANGUAGE CUDA)
