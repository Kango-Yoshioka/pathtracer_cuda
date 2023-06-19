//
// Created by kango on 2023/06/17.
//
#include "Object/Body/Body.cuh"
#include "Object/Geometry/Sphere.cuh"
#include "Renderer/Camera/Camera.cuh"
#include "Renderer/Renderer.cuh"
#include "Renderer/Scene.cuh"
#include "vector"
#include <iostream>
#include <windows.h>

__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}

void func1() {
    std::cout << "Hello, World!" << std::endl;

    cuda_hello<<<1, 1>>>();

    cudaDeviceSynchronize();

    const Color offWhite(1.0, 0.97, 0.93);

    Sphere worldSphere(1e10, Eigen::Vector3d(0, 0, 0));
    const double room_r = 1e5;
    Sphere roomSpheres[] = {
            Sphere(room_r, Eigen::Vector3d{room_r - 15, 0, 0}),
            Sphere(room_r, Eigen::Vector3d{-(room_r - 15), 0, 0}),
            Sphere(room_r, Eigen::Vector3d{0, 0, room_r - 30}),
            Sphere(room_r, Eigen::Vector3d{0, room_r - 10, 0}),
            Sphere(room_r, Eigen::Vector3d{0, -(room_r - 10), 0})
    };
    Sphere lightSphere(1.0, Eigen::Vector3d(0, 8.9, 7));

    Sphere sphere(3.0, Eigen::Vector3d{0, -7, -6});
    Sphere sphere2(2.0, Eigen::Vector3d{-8, -8, -4});
    Sphere sphere3(2.0, Eigen::Vector3d{7, -8, -5});

    Body world(0.0, Material(M_DIFFUSE, Color::Zero(), 0.0), worldSphere);
    const double room_kd = 0.6;
    Body room[] = {
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 0.01, 0.01), room_kd), roomSpheres[0]),
            Body(0.0, Material(M_DIFFUSE, Color(0.01, 1.0, 0.01), room_kd), roomSpheres[1]),
            Body(0.0, Material(M_DIFFUSE, offWhite, room_kd), roomSpheres[2]),
            Body(0.0, Material(M_DIFFUSE, offWhite, room_kd), roomSpheres[3]),
            Body(0.0, Material(M_DIFFUSE, offWhite, room_kd), roomSpheres[4]),
    };
    Body light(1000.0, Material(M_ZERO, Color(1, 1, 1)), lightSphere);
    Body body(0.0, Material(Color(0.3, 0.92, 0.95), 1.0, 0.0, 0.0), sphere);
    Body body2(0.0, Material(Color(0.6, 0.7, 0.5), 0.01, 0.9, 0.0), sphere2);
    Body body3(0.0, Material(codeToColor("#5900ff"), 0.01, 0.1, 0.8), sphere3);
    Body body4(0.0, Material(Color(0.76, 0.67, 1.0), 0.6, 0.3, 0.001), Sphere(1.5, Eigen::Vector3d{3, -8.5, 0}));
    Body body5(0.0, Material(codeToColor("#FFC800"), 0.8, 0.1, 0.001), Sphere(1.5, Eigen::Vector3d{-4, -8.5, -1}));
    Body body6(0.0, Material(codeToColor("#bdd458"), 0.3, 0.7, 0.0), Sphere(2.0, Eigen::Vector3d{-5.0, -8.0, -8.0}));
    Body body7(0.0, Material(codeToColor("#FFFF88"), 0.3, 0.1, 0.6), Sphere(2.0, Eigen::Vector3d{-0.5, -10.0, 5.0}));
    std::vector<Body> bodies{world, light, body, body2, body3, body4, body5, body6, body7};
    for(auto & i : room) {
        bodies.push_back(i);
    }

    const Eigen::Vector3d camOrg{0, -5, 15.0};
    const Camera camera(
            camOrg,
            body.getSphere().center - camOrg,
            1080, 16.0 / 9.0, 40, 1.0, (body.getSphere().center - camOrg).norm(), 1.5, 20
    );

    Scene scene(bodies.size(), camera, bodies.data(), Color::Zero());

    scene.printSceneInfo();

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    cudaDeviceSynchronize();

    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);

    auto image = generateImageWithGPU(scene, 1e4).apply_reinhard_extended_tone_mapping().apply_bilateral_filter(5, 150, 10).apply_gamma_correction();

    QueryPerformanceCounter(&end);

    const double time = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
    std::cout << "Generation time\t" << time << " [sec]" << std::endl;

    image.show();
    image.save("../../Results/sample.png");

    cudaDeviceReset();
}

void func2() {
    const auto floor_color = codeToColor("#f9c89b");
    const double room_r = 1e5;
    const double room_kd = 0.8;
    Sphere roomSpheres[] = {
            Sphere(room_r, Eigen::Vector3d{room_r - 40, 0, 0}),
            Sphere(room_r, Eigen::Vector3d{-(room_r - 40), 0, 0}),
            Sphere(room_r, Eigen::Vector3d{0, 0, room_r - 100}),
            Sphere(room_r, Eigen::Vector3d{0, room_r - 50, 0}),
            Sphere(room_r, Eigen::Vector3d{0, -(room_r - 50), 0})
    };

    Body room[] = {
            Body(0.0, Material(M_DIFFUSE, codeToColor("#2f5d50"), room_kd), roomSpheres[0]),
            Body(0.0, Material(M_DIFFUSE, codeToColor("#00a3af"), room_kd), roomSpheres[1]),
            Body(0.0, Material(M_DIFFUSE, floor_color, room_kd), roomSpheres[2]),
            Body(0.0, Material(M_DIFFUSE, floor_color, room_kd), roomSpheres[3]),
            Body(0.0, Material(M_DIFFUSE, floor_color, room_kd), roomSpheres[4]),
    };

    std::vector<Body> bodies {
            Body(25, Material(codeToColor("#e597b2"), 1.0, 0.0, 0.0), Sphere(10, Eigen::Vector3d(0, 40, 0))),
            Body(0.0, Material(Color(0.75, 0.25, 0.25), 0.8, 0.0, 0.0), Sphere(12, Eigen::Vector3d(15, -38, -30))),
            Body(0.0, Material(codeToColor("#e5e4e6"), 0.8, 0.0, 0.0), Sphere(15, Eigen::Vector3d(-18, -35, -45))),
            Body(0.0, Material(codeToColor("#b0c4de"), 0.3, 0.5, 0.0), Sphere(8, Eigen::Vector3d(-3, -42, 5))),
            Body(0.0, Material(codeToColor("#c75595"), 0.6, 0.01, 0.0), Sphere(8, Eigen::Vector3d(-15, -42, -10))),
            Body(0.0, Material(codeToColor("#bdb76b"), 0.3, 0.2, 0.2), Sphere(9, Eigen::Vector3d(12, -41, -3))),
    };

    for(int i = 0; i < 5; i++) {
        bodies.emplace_back(room[i]);
    }

    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for(int i = 1; i < 6; ++i) sum += bodies[i].getSphere().center;
    const Eigen::Vector3d camOrg{0, 5, 100.0};
    const Eigen::Vector3d camdir = sum / 5.0 - camOrg;
    const Camera camera(
            camOrg,
            camdir,
            1920, 9.0 / 16.0, 50, 1.0, camdir.norm(), 0.5, 20
    );

    Scene scene(bodies.size(), camera, bodies.data(), Color::Zero());

    scene.printSceneInfo();

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    cudaDeviceSynchronize();

    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);

    auto image = generateImageWithGPU(scene, 1e4).apply_reinhard_extended_tone_mapping().apply_bilateral_filter(5, 150, 10).apply_gamma_correction();

    QueryPerformanceCounter(&end);

    const double time = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
    std::cout << "Generation time\t" << time << " [sec]" << std::endl;

    // image.show();
    image.save("../../Results/sample2.png");

    cudaDeviceReset();
}

int main() {
    func2();
    return 0;
}
