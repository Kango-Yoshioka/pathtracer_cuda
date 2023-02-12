//
// Created by kango on 2023/01/01.
//

#ifndef CUDATEST_RAY_CUH
#define CUDATEST_RAY_CUH


#include "Eigen/Dense"


enum SIDE {
    S_FRONT = 1,
    S_BACK = -1
};

struct Ray {
    Eigen::Vector3d org{}, dir{};

    __host__ __device__
    Ray() = default;

    __host__ __device__
    Ray(Eigen::Vector3d org, Eigen::Vector3d dir) :
            org(std::move(org)), dir(std::move(dir.normalized())) {}
};

struct RayHit {
    int idx{};
    SIDE side{};
    double t{};
    Eigen::Vector3d normal;

    __host__ __device__
    RayHit() = default;
};


#endif //CUDATEST_RAY_CUH
