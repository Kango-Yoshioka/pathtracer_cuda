//
// Created by kango on 2023/01/04.
//

#include "Camera.cuh"

Camera::Camera(const Eigen::Vector3d &org, const Eigen::Vector3d &dir, const int &resolutionHeight, double aspectRatio, double filmHeight, double focalLength)
    : Ray(org, dir.normalized()), Film(resolutionHeight, aspectRatio, filmHeight), focalLength(focalLength) {
    up = Eigen::Vector3d{0, 1, 0} * filmHeight;
    right = dir.normalized().cross(up).normalized() * filmHeight * aspectRatio;
}

__host__ __device__
void Camera::filmView(const unsigned int &p_x, const unsigned int &p_y, Ray &out_ray) const {
    const auto pixelLocalPos = pixelLocalPosition(p_x, p_y);
    out_ray.org = org;
    out_ray.dir = (right * (pixelLocalPos.x() - 0.5) + up * (0.5 - pixelLocalPos.y()) + dir * focalLength).normalized();
}