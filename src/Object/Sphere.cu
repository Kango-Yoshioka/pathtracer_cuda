//
// Created by kango on 2023/01/02.
//

#include "Sphere.cuh"

__device__ __host__
Sphere::Sphere(const double &radius, Eigen::Vector3d center) : radius(radius), center(std::move(center)) {}

__device__ __host__
Eigen::Vector3d Sphere::getNormal(const SIDE &side, const Eigen::Vector3d &p) const {
    return side * (p - center).normalized();
}

__device__ __host__
SIDE Sphere::getHitSide(const Eigen::Vector3d &p, const Ray &ray) const {
    return ray.dir.dot(getNormal(S_FRONT, p)) < 0 ? S_FRONT : S_BACK;
}

__device__ __host__
bool Sphere::hit(const Ray &ray, RayHit &rayHit) const {
    const double b = (ray.org - center).dot(ray.dir);
    const double c = (ray.org - center).squaredNorm() - radius * radius;
    const double discriminant = b * b - c;

    if (discriminant < 0.0) return false;

    const Eigen::Array2d distances{(-b - sqrt(discriminant)),
                                   (-b + sqrt(discriminant))
    };

    if((distances < 1e-6).all()) return false;

    rayHit.t = distances[0] > 1e-6 ? distances[0] : distances[1];
    const auto p = ray.org + rayHit.t * ray.dir;
    rayHit.side = getHitSide(p, ray);
    rayHit.normal = getNormal(rayHit.side, p).normalized();
    return true;
}
