set(CMAKE_VERBOSE_MAKEFILE 1)

cmake_minimum_required(VERSION 3.24)
project(pathtracer_cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(Image)
add_subdirectory(Object)
add_subdirectory(Renderer)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} PUBLIC image)
target_link_libraries(${PROJECT_NAME} PUBLIC body)
target_link_libraries(${PROJECT_NAME} PUBLIC renderer)
