//
// Created by kango on 2023/01/06.
//

#ifndef LINKTEST_BODY_CUH
#define LINKTEST_BODY_CUH


#include "../Sphere.cuh"
#include "../Material/Material.cuh"
#include "Eigen/Dense"

class Body {
protected:
    double emission;
    Material material;
    Sphere sphere;
public:
    Body(const double &emission, Material material, Sphere sphere)
        : emission(emission), material(std::move(material)), sphere(std::move(sphere)) {}

    [[nodiscard]] __host__ __device__ double getEmission() const;

    [[nodiscard]]  __host__ __device__ const Material &getMaterial() const;

    [[nodiscard]]  __host__ __device__ const Sphere &getSphere() const;

    [[nodiscard]]  __host__ __device__ Eigen::Vector3d getNormal(const SIDE &side, const Eigen::Vector3d &p) const;

    __host__ __device__ bool hit(const Ray &ray, RayHit &rayHit) const;
};


#endif//LINKTEST_BODY_CUH
