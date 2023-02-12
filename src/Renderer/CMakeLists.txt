set(CMAKE_VERBOSE_MAKEFILE 1)
cmake_policy(SET CMP0076 NEW)

cmake_minimum_required(VERSION 3.24)
project(renderer_lib
        VERSION 1.0.0
        DESCRIPTION "Object library"
        LANGUAGES CUDA)


add_library(renderer STATIC Renderer.cu Renderer.cuh)

add_subdirectory(Camera)

target_link_libraries(renderer PUBLIC camera)
target_link_libraries(renderer PUBLIC image)
target_link_libraries(renderer PUBLIC object)

target_include_directories(renderer PRIVATE ../../third_party/eigen-3.4.0)
target_include_directories(renderer PRIVATE ../Object)

# cuda-samples
#---------------------------------------
include(FetchContent)
FetchContent_Declare(
        cuda-samples
        GIT_REPOSITORY "https://github.com/NVIDIA/cuda-samples.git"
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(cuda-samples)
FetchContent_GetProperties(cuda-samples SOURCE_DIR CUDA_SAMPLES_INCLUDE_DIR)
message("-- Found cuda-sample ${CUDA_SAMPLES_INCLUDE_DIR}")
target_include_directories(renderer PUBLIC ${CUDA_SAMPLES_INCLUDE_DIR}/Common)

set_target_properties(renderer
        PROPERTIES
        VERSION ${PROJECT_VERSION})

set_target_properties(renderer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(renderer
        PROPERTIES
        LINKER_LANGUAGE CUDA)