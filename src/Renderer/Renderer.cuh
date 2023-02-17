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

__device__ __forceinline__ bool hitScene(const Scene *scene, const Ray &ray, RayHit &hit);

__device__ __forceinline__ void computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v);

__device__ __forceinline__ void diffuseSample(const Eigen::Vector3d &normal, Ray &ray, const Eigen::Vector3d &incidentPoint, const double2 &rands);

__device__ __forceinline__ void specularSample(const Eigen::Vector3d &normal, Ray &ray, const Eigen::Vector3d &incidentPoint);

__device__ __forceinline__ void refractSample(const Eigen::Vector3d &normal, Ray &ray, const Eigen::Vector3d &incidentPoint);

__global__ void sceneInitialize(Scene *d_scene, Body *d_body);

__global__ void writeToPixels(Color *out_pixels, Scene *scene, unsigned int samplesPerPixel, curandState *states);

__host__ Image generateImageWithGPU(const Scene &scene, const unsigned int &samplesPerPixel);

__device__ __forceinline__ void pathTraceGPU(Color &radiance, const Scene *scene, Ray &ray, curandState &state, PATH_TRACE_FLAG &flag);

__device__ __forceinline__ double generateRandom(curandState &state);

#endif //CUDATEST_RENDERER_CUH
