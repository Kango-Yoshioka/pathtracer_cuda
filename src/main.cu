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
    Sphere sphere(1.0, Eigen::Vector3d::Zero());
    Sphere sphere2(0.60, Eigen::Vector3d{0.5, 0, 2});

    const Camera camera(
            Eigen::Vector3d{0, 0, 8},
            Eigen::Vector3d{0, 0, -1}.normalized(),
            720, 4.0 / 3.0, 3.0, 1.0
    );

    Body world(0.0, Material(M_DIFFUSE, Color::Zero(), 0.0), worldSphere);
    Body room[6] = {
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 0.01, 0.01), 0.5), roomSpheres[0]),
            Body(0.0, Material(M_DIFFUSE, Color(0.01, 1.0, 0.01), 0.5), roomSpheres[1]),
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 1.0, 1.0), 0.6), roomSpheres[2]),
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 1.0, 1.0), 0.8), roomSpheres[3]),
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 1.0, 1.0), 0.6), roomSpheres[4]),
            Body(0.0, Material(M_DIFFUSE, Color(1.0, 1.0, 1.0), 0.6), roomSpheres[5])
    };
    Body light(100.0, Material(M_ZERO, Color(1, 1, 1)), lightSphere);
    Body body(0.0, Material(Color(0.3, 0.92, 0.95), 0.1, 0.7, 0.0), sphere);
    Body body2(0.0, Material(Color(0.6, 0.7, 0.5), 0.1, 0.7, 0.0), sphere2);
    std::vector<Body> bodies{world, light, body, body2};
    for(auto & i : room) {
        bodies.push_back(i);
    }

    Scene scene(bodies.size(), camera, bodies.data(), Color::Zero());

    scene.printSceneInfo();

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);

    generateImageWithGPU(scene, static_cast<int>(pow(2, 12)));

    QueryPerformanceCounter(&end);

    const double time = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
    std::cout << "Generation time\t" << time << " [sec]" << std::endl;

    return 0;
}
