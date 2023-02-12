set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(object_lib
        VERSION 1.0.0
        DESCRIPTION "Object library"
        LANGUAGES CUDA)

add_subdirectory(Material)
add_subdirectory(Body)

add_library(object STATIC Sphere.cu)

target_link_libraries(object PUBLIC image body material)

target_include_directories(object PUBLIC ../../third_party/eigen-3.4.0)

set_target_properties(object
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(object
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(object
        PROPERTIES
        LINKER_LANGUAGE CUDA)
