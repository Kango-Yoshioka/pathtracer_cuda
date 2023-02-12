//
// Created by kango on 2023/01/02.
//

#ifndef CUDATEST_SPHERE_CUH
#define CUDATEST_SPHERE_CUH

#include "../Renderer/Ray.cuh"
#include "Eigen/Dense"

class Sphere {
public:
    double radius;
    Eigen::Vector3d center;

    __device__ __host__
    Sphere(const double &radius, Eigen::Vector3d center);

    __device__ __host__ __forceinline__
    SIDE getHitSide(const Eigen::Vector3d &p, const Ray &ray) const;

    __device__ __host__ __forceinline__
    Eigen::Vector3d getNormal(const SIDE &side, const Eigen::Vector3d &p) const;

    __device__ __host__ __forceinline__
    bool hit(const Ray &ray, RayHit &rayHit) const;
};


#endif //CUDATEST_SPHERE_CUH
