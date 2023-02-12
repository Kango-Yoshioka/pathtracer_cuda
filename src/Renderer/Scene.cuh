//
// Created by kango on 2023/01/06.
//

#ifndef LINKTEST_SCENE_CUH
#define LINKTEST_SCENE_CUH


#include <utility>

#include "../Image/Image.cuh"
#include "../Object/Body/Body.cuh"
#include "Camera/Camera.cuh"

struct Scene {
    unsigned int bodiesSize;
    Camera camera;
    Body* bodies;
    Color backgroundColor;

    Scene(const unsigned int &bodiesSize, Camera camera, Body* bodies, Color  backgroundColor)
        : bodiesSize(bodiesSize), camera(std::move(camera)), bodies(bodies), backgroundColor(std::move(backgroundColor)) {}

    __host__ __device__ void printSceneInfo() {
        printf("### Scene Information ###\n");
        printf("backgroundColor:\t(%lf, %lf, %lf)\n", backgroundColor.x(), backgroundColor.y(), backgroundColor.z());

        printf("## Camera ##\n");
        printf("org:\t(%lf, %lf, %lf)\n", camera.org.x(), camera.org.y(), camera.org.z());
        printf("dir:\t(%lf, %lf, %lf)\n", camera.dir.x(), camera.dir.y(), camera.dir.z());
        printf("right:\t(%lf, %lf, %lf)\n", camera.right.x(), camera.right.y(), camera.right.z());
        printf("up:\t(%lf, %lf, %lf)\n", camera.up.x(), camera.up.y(), camera.up.z());
        printf("focalLength:\t%lf\n", camera.focalLength);
        printf("## ----- ##\n");

        printf("bodiesSize:\t%d\n", bodiesSize);
        printf("## Bodies ##\n");
        for(int i = 0; i < bodiesSize; i++) {
            printf("-- idx:\t%d --\n", i);
            printf("emission:\t%lf\n", bodies[i].getEmission());

            const Material material = bodies[i].getMaterial();
            printf("# material #\n");
            printf("color:\t(%lf, %lf, %lf)\n", material.getColor().x(), material.getColor().y(), material.getColor().z());
            printf("kd, ks, kt:\t%lf, %lf, %lf\n", material.getKd(), material.getKs(), material.getKt());
            printf("# -------- #\n");

            const Sphere sphere = bodies[i].getSphere();
            printf("# sphere #\n");
            printf("radius:\t%lf\n", sphere.radius);
            printf("center:\t(%lf, %lf, %lf)\n", sphere.center.x(), sphere.center.y(), sphere.center.z());
            printf("# ------ #\n");

            printf("-- +++++ --\n");
        }
        printf("## ------ ##\n");
    }
};


#endif//LINKTEST_SCENE_CUH
