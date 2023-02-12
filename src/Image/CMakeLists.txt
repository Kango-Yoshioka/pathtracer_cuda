set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(image_lib
        VERSION 1.0.0
        DESCRIPTION "Image library"
        LANGUAGES CUDA)

cmake_policy(SET CMP0076 NEW)

add_library(image STATIC Image.cu)

target_include_directories(image
        PUBLIC
        ../../third_party/eigen-3.4.0
        ../../third_party/stbi)

set_target_properties(image
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(image
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(image
        PROPERTIES
        LINKER_LANGUAGE CUDA)
