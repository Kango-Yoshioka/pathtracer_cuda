set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(material_lib
        VERSION 1.0.0
        DESCRIPTION "Material library"
        LANGUAGES CUDA)


add_library(material STATIC Material.cu)

target_link_libraries(material PUBLIC image)

set_target_properties(material
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(material
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(material
        PROPERTIES
        LINKER_LANGUAGE CUDA)