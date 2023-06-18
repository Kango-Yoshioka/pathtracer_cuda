set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(image_lib
        VERSION 1.0.0
        DESCRIPTION "Image library"
        LANGUAGES CXX)

add_library(image STATIC Image.cpp)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(image PUBLIC Eigen3::Eigen)

find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(image PUBLIC opencv_ml opencv_dnn opencv_core opencv_flann opencv_imgcodecs opencv_highgui)

set_target_properties(image
        PROPERTIES
        VERSION ${PROJECT_VERSION})
