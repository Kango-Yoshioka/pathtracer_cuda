//
// Created by kango on 2023/01/04.
//

#include "Camera.cuh"

Camera::Camera(const Eigen::Vector3d &org, const Eigen::Vector3d &dir, const int &resolutionHeight, double aspectRatio, double verticalFoV, double aperture, double focusDist)
    : Ray(org, dir), lensRadius(aperture / 2.0), focusDist(focusDist) {
    // 度数法からradianに変換
    const auto theta = verticalFoV * EIGEN_PI / 180.0;
    auto filmHeight = 2 * tan(theta / 2.0);
    right = dir.cross(Eigen::Vector3d{0, 1, 0}).normalized() * filmHeight * aspectRatio * focusDist;
    up = right.cross(dir).normalized() * filmHeight * focusDist;
    film = Film(resolutionHeight, aspectRatio, filmHeight);
}

__host__ __device__
void Camera::filmView(const unsigned int &p_x, const unsigned int &p_y, Ray &out_ray, const Eigen::Vector4d &rand) const {
    const auto pixelLocalPos = film.pixelLocalPosition(p_x, p_y, Eigen::Vector2d{rand.w(), rand.x()});
    const double theta = 2 * EIGEN_PI * rand.y();
    const double r = lensRadius * rand.z();
    const Eigen::Vector3d offset = r * (right.normalized() * cos(theta) + up.normalized() * sin(theta));
    out_ray.org = org + offset;
    out_ray.dir = (right * (pixelLocalPos.x() - 0.5) + up * (0.5 - pixelLocalPos.y()) + dir * focusDist - offset).normalized();
}