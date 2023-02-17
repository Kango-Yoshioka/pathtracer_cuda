//
// Created by kango on 2023/01/02.
//

#ifndef CUDATEST_CAMERA_CUH
#define CUDATEST_CAMERA_CUH


#include "../Film.cuh"
#include "../Ray.cuh"

class Camera : public Ray {
public:
    double focalLength, focusDist, fNumber, camToLensDist, sensitivity;
    Eigen::Vector3d right, up, lensPos;
    Film film;

    Camera(const Eigen::Vector3d &org, const Eigen::Vector3d &dir, const int &resolutionHeight, double aspectRatio, double verticalFoV, double focalLength, double focusDist, double fNumber, double sensitivity);

    __device__ void filmView(const unsigned int &p_x, const unsigned int &p_y, Ray &out_ray, double &weight, const double4 &rand) const;
};


#endif //CUDATEST_CAMERA_CUH
