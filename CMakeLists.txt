set(CMAKE_VERBOSE_MAKEFILE 1)

cmake_minimum_required(VERSION 3.24)
project(pathtracer_cuda LANGUAGES CUDA CXX)

add_subdirectory(src)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)

# CUDA
#---------------------------------------
if(CMAKE_CUDA_COMPILER_LOADED)
    message("-- CUDA --")
    message("Compiler: ${CMAKE_CUDA_COMPILER}")
    message("Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

endif()

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
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_SAMPLES_INCLUDE_DIR}/Common)
