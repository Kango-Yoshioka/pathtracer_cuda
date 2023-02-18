set(CMAKE_VERBOSE_MAKEFILE 1)


cmake_minimum_required(VERSION 3.24)
project(pathtracer_cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)

add_executable(${PROJECT_NAME} src/main.cu)

add_subdirectory(src/Image)
add_subdirectory(src/Object/Body)
add_subdirectory(src/Object/Geometry)
add_subdirectory(src/Object/Material)
add_subdirectory(src/Renderer)

target_link_libraries(${PROJECT_NAME} PUBLIC image body geometry material renderer curand)
target_include_directories(${PROJECT_NAME} PRIVATE third_party/eigen-3.4.0)

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

#for OpenMP
find_package(OpenMP)
if(OPENMP_CXX_FOUND)
    # add flags for OpenMP
    message("-- Found openmp >> ${OpenMP_FOUND} ${OpenMP_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

