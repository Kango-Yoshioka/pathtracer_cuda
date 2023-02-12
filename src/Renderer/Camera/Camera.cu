//
// Created by kango on 2023/01/04.
//

#include "Camera.cuh"

Camera::Camera(const Eigen::Vector3d &org, const Eigen::Vector3d &dir, const int &resolutionHeight, double aspectRatio, double verticalFoV, double focalLength)
    : Ray(org, dir), focalLength(focalLength) {
    // 度数法からradianに変換
    const auto theta = verticalFoV * EIGEN_PI / 180.0;
    auto filmHeight = 2 * tan(theta / 2.0);
    right = dir.cross(Eigen::Vector3d{0, 1, 0}).normalized() * filmHeight * aspectRatio;
    up = right.cross(dir).normalized() * filmHeight;
    film = Film(resolutionHeight, aspectRatio, filmHeight);
}

__host__ __device__
void Camera::filmView(const unsigned int &p_x, const unsigned int &p_y, Ray &out_ray) const {
    const auto pixelLocalPos = film.pixelLocalPosition(p_x, p_y);
    out_ray.org = org;
    out_ray.dir = (right * (pixelLocalPos.x() - 0.5) + up * (0.5 - pixelLocalPos.y()) + dir * focalLength).normalized();
}