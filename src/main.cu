#include "Object/Body/Body.cuh"
#include "Object/Sphere.cuh"
#include "Renderer/Camera/Camera.cuh"
#include "Renderer/Renderer.cuh"
#include "Renderer/Scene.cuh"
#include "vector"
#include <iostream>
#include <windows.h>

__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}


int main() {
    std::cout << "Hello, World!" << std::endl;

    cuda_hello<<<1, 1>>>();

    cudaDeviceSynchronize();

    const Color offWhite(1.0, 0.97, 0.93);

    Sphere worldSphere(1e10, Eigen::Vector3d(0, 0, 0));
    const double room_r = 1e5;
    Sphere roomSpheres[6] = {
            Sphere(room_r, Eigen::Vector3d{room_r - 10, 0, 0}),
            Sphere(room_r, Eigen::Vector3d{-(room_r - 10), 0, 0}),
            Sphere(room_r, Eigen::Vector3d{0, 0, room_r - 10}),
            Sphere(room_r, Eigen::Vector3d{0, 0, -(room_r - 10)}),
            Sphere(room_r, Eigen::Vector3d{0, room_r - 10, 0}),
            Sphere(room_r, Eigen::Vector3d{0, -(room_r - 10), 0})
    };
    Sphere lightSphere(1.0, Eigen::Vector3d(0, 7, 7));
    Sphere sphere(3.0, Eigen::Vector3d{0, -7, -6});
    Sphere sphere2(2.0, Eigen::Vector3d{-8, -8, -4});
    Sphere sphere3(2.0, Eigen::Vector3d{7, -8, -5});

    Body world(0.0, Material(M_DIFFUSE, Color::Zero(), 0.0), worldSphere);
    Body room[6] = {
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 0.01, 0.01), 0.5), roomSpheres[0]),
            Body(0.0, Material(M_DIFFUSE, Color(0.01, 1.0, 0.01), 0.5), roomSpheres[1]),
            Body(0.0, Material(M_DIFFUSE, offWhite, 0.6), roomSpheres[2]),
            Body(0.0, Material(M_DIFFUSE, offWhite, 0.6), roomSpheres[3]),
            Body(0.0, Material(M_DIFFUSE, offWhite, 0.6), roomSpheres[4]),
            Body(0.0, Material(M_DIFFUSE, offWhite, 0.6), roomSpheres[5])
    };
    Body light(100.0, Material(M_ZERO, Color(1, 1, 1)), lightSphere);
    Body body(0.0, Material(Color(0.3, 0.92, 0.95), 1.0, 0.0, 0.0), sphere);
    Body body2(0.0, Material(Color(0.6, 0.7, 0.5), 0.01, 0.9, 0.0), sphere2);
    Body body3(0.0, Material(Color(1.0, 1.0, 1.0), 0.01, 0.7, 0.001), sphere3);
    Body body4(0.0, Material(Color(0.76, 0.67, 1.0), 0.6, 0.3, 0.001), Sphere(1.5, Eigen::Vector3d{3, -8.5, 0}));
    Body body5(0.0, Material(codeToColor("#FFC800"), 0.8, 0.1, 0.001), Sphere(1.5, Eigen::Vector3d{-4, -8.5, -1}));
    Body body6(0.0, Material(codeToColor("#bdd458"), 0.3, 0.7, 0.0), Sphere(2.0, Eigen::Vector3d{-5.0, -8.0, -8.0}));
    Body body7(0.0, Material(codeToColor("#FFFFFF"), 0.8, 0.2, 0.0), Sphere(2.0, Eigen::Vector3d{-0.5, -10.0, 5.0}));
    std::vector<Body> bodies{world, light, body, body2, body3, body4, body5, body6, body7};
    for(auto & i : room) {
        bodies.push_back(i);
    }

    const Eigen::Vector3d camOrg{0, -8.0, 9};
    const Camera camera(
            camOrg,
            sphere.center - camOrg,
            720, 16.0 / 9.0, 40, 0.8, (sphere.center - camOrg).norm(), 2.0
    );

    Scene scene(bodies.size(), camera, bodies.data(), Color::Zero());

    scene.printSceneInfo();

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);

    auto image = generateImageWithGPU(scene, static_cast<int>(pow(2, 10)));

    QueryPerformanceCounter(&end);

    const double time = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
    std::cout << "Generation time\t" << time << " [sec]" << std::endl;

    image.generatePNG("../Results/sampleGPU");
    image.generateCSV("../Results/sampleGPU");
    return 0;
}
