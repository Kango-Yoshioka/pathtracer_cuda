//
// Created by kango on 2023/01/06.
//

#include "Body.cuh"

__host__ __device__
double Body::getEmission() const {
    return emission;
}

__host__ __device__
const Material &Body::getMaterial() const {
    return material;
}

__host__ __device__
const Sphere &Body::getSphere() const {
    return sphere;
}

__host__ __device__
Eigen::Vector3d Body::getNormal(const SIDE &side, const Eigen::Vector3d &p) const {
    return sphere.getNormal(side, p);
}

__host__ __device__
bool Body::hit(const Ray &ray, RayHit &rayHit) const {
    return sphere.hit(ray, rayHit);
}
