set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(camera_lib
        VERSION 1.0.0
        DESCRIPTION "Object library"
        LANGUAGES CUDA)

add_library(camera STATIC Camera.cu)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(camera PUBLIC Eigen3::Eigen)

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set_target_properties(camera
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(camera
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(camera
        PROPERTIES
        LINKER_LANGUAGE CUDA)
