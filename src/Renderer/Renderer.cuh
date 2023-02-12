//
// Created by kango on 2023/01/02.
//

#ifndef CUDATEST_RENDERER_CUH
#define CUDATEST_RENDERER_CUH


#include "../Image/Image.cuh"
#include "../Object/Sphere.cuh"
#include "Camera/Camera.cuh"
#include "Scene.cuh"
#include <curand_kernel.h>
#include <random>

enum PATH_TRACE_FLAG {
    PATH_TRACE_CONTINUE = true,
    PATH_TRACE_TERMINATE = false
};

__host__ void generateImageWithCPU(const Scene &scene);

__device__ __host__ __forceinline__ bool hitScene(const Scene *scene, const Ray &ray, RayHit &rayHit);

__host__ __device__ __forceinline__ void computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v);

__host__ __device__ __forceinline__ void diffuseSample(const Eigen::Vector3d &normal, Ray &out_ray, const Eigen::Vector3d &incidentPoint, const double2 &rands);

__device__ __forceinline__ void specularSample(const Eigen::Vector3d &normal, const Ray &in_ray, Ray &out_ray, const Eigen::Vector3d &incidentPoint);

__device__ __forceinline__ void refractSample(const Eigen::Vector3d &normal, const Ray &in_ray, Ray &out_ray, const Eigen::Vector3d &incidentPoint);

__global__ void sceneInitialize(Scene *d_scene, Body *d_body);

__global__ void writeToPixels(Color *out_pixels, Scene *scene, unsigned int samplesPerPixel, curandState *states);

__host__ Image generateImageWithGPU(const Scene &scene, const unsigned int &samplesPerPixel);

__host__ void pathTraceCPU(const Color &in_radiance, Color &out_radiance, const Scene *scene, const Ray &in_ray, Ray &out_ray, PATH_TRACE_FLAG &flag, std::default_random_engine &engine, std::uniform_real_distribution<> &dist);

__device__ __forceinline__ void pathTraceGPU(const Color &in_radiance, Color &out_radiance, const Scene *scene, const Ray &in_ray, Ray &out_ray, curandState &state, PATH_TRACE_FLAG &flag);

__device__ __forceinline__ double generateRandom(curandState &state);

__device__ void pathTraceGPU2(const int &pixelIdx, const Color &in_radiance, Color &out_radiance, const Scene *scene, const Ray &in_ray, Ray &out_ray, curandState &state, PATH_TRACE_FLAG &flag);

#endif //CUDATEST_RENDERER_CUH
