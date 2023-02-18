set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(geometry_lib
        VERSION 1.0.0
        DESCRIPTION "Geometry library"
        LANGUAGES CUDA)


add_library(geometry STATIC sphere.cu)

target_include_directories(geometry PUBLIC ../../Renderer ../../../third_party/eigen-3.4.0)

set_target_properties(geometry
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(geometry
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(geometry
        PROPERTIES
        LINKER_LANGUAGE CUDA)