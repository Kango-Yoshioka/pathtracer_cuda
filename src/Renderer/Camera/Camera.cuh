//
// Created by kango on 2023/01/02.
//

#ifndef CUDATEST_CAMERA_CUH
#define CUDATEST_CAMERA_CUH


#include "../Film.cuh"
#include "../Ray.cuh"

class Camera : public Ray, public Film {
public:
    /// カメラとフィルムの距離
    double focalLength;
    Eigen::Vector3d right, up;

    Camera(const Eigen::Vector3d &org, const Eigen::Vector3d &dir, const int &resolutionHeight, double aspectRatio, double filmHeight, double focalLength);

    __host__ __device__
    void filmView(const unsigned int &p_x, const unsigned int &p_y, Ray &out_ray) const;
};


#endif //CUDATEST_CAMERA_CUH
