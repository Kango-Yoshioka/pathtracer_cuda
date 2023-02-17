//
// Created by kango on 2023/01/02.
//

#include "Renderer.cuh"
#include "Scene.cuh"
#include "helper_cuda.h"
#include <iostream>
#include <windows.h>

/// Random number Generator Functions
__global__ void curandInitInRenderer(const Scene *scene, curandState* state, unsigned long seed) {
    const Eigen::Vector2i p(
            (blockIdx.x * blockDim.x) + threadIdx.x,
            (blockIdx.y * blockDim.y) + threadIdx.y
    );

    if(p.x() >= scene->camera.film.resolution.x() || p.y() >= scene->camera.film.resolution.y()) return;

    const unsigned int pixelIdx = p.x() + (p.y() * scene->camera.film.resolution.x());
    curand_init(seed, pixelIdx, 0, &state[pixelIdx]);
}

__device__ __forceinline__
double generateRandom(curandState &state) {
    curandState localState = state;
    const double RANDOM = curand_uniform(&localState);
    state = localState;
    return RANDOM;
}
////////////////////

__device__ __forceinline__
bool hitScene(const Scene *scene, const Ray &ray, RayHit &hit) {
    hit.t = DBL_MAX;
    hit.idx = -1;
    for (int i = 0; i < scene->bodiesSize; i++) {
        RayHit _hit;
        if (scene->bodies[i].hit(ray, _hit) && _hit.t < hit.t) {
            hit.t = _hit.t;
            hit.idx = i;
            hit.side = _hit.side;
            hit.normal = _hit.normal;
        }
    }
    return hit.idx != -1;
}

__global__
void writeToPixels(Color *out_pixels, Scene *scene, unsigned int samplesPerPixel, curandState *states) {
    const Eigen::Vector2i p(
            (blockIdx.x * blockDim.x) + threadIdx.x,
            (blockIdx.y * blockDim.y) + threadIdx.y
    );
    if(p.x() >= scene->camera.film.resolution.x() || p.y() >= scene->camera.film.resolution.y()) return;
    const unsigned int pixelIdx = p.x() + (p.y() * scene->camera.film.resolution.x());

    curandState state = states[pixelIdx];
    Color accumulate_radiance = Color::Zero();

    for(int i = 0; i < samplesPerPixel; i++) {
        const double4 rand = make_double4(generateRandom(state), generateRandom(state), generateRandom(state), generateRandom(state));
        Ray ray; double weight;
        Color radiance = Color::Ones();
        PATH_TRACE_FLAG pathTraceFlag;
        scene->camera.filmView(p.x(), p.y(), ray, weight, rand);
        do {
            pathTraceGPU(radiance, scene, ray, state, pathTraceFlag);
        } while (pathTraceFlag);
        accumulate_radiance += weight * radiance;
    }

     out_pixels[pixelIdx] = accumulate_radiance / static_cast<double>(samplesPerPixel);
}

__global__
void sceneInitialize(Scene *d_scene, Body *d_body) {
    d_scene->bodies = d_body;
}

__host__
Image generateImageWithGPU(const Scene &scene, const unsigned int &samplesPerPixel) {
    Scene* d_scene;
    Body* d_body;
    Color* d_pixels;
    curandState* d_state;

    /// information of camera ///
    std::cout << "-- Camera --" << std::endl;
    std::cout << "Resolution\t(" << scene.camera.film.resolution.x() << ", " << scene.camera.film.resolution.y() << ")" << std::endl;
    std::cout << "------------" << std::endl;

    /// Scene initialize ///
    Image h_image(scene.camera.film.resolution.x(), scene.camera.film.resolution.y());
    std::cout << "Pixel size:\t" << h_image.getWidth() * h_image.getHeight() << std::endl;

    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
    /**
     * 構造体内に配列がある場合、cudaMemcpyをしても中身まではコピーされず、エラーとなるので、
     * 構造体内の配列は別途でGPUに送る必要あり。
     */
    checkCudaErrors(cudaMalloc((void**)&d_body, sizeof(Body) * scene.bodiesSize));
    checkCudaErrors(cudaMalloc((void**)&d_pixels, sizeof(Color) * h_image.getWidth() * h_image.getHeight()));

    checkCudaErrors(cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_body, scene.bodies, sizeof(Body) * scene.bodiesSize, cudaMemcpyHostToDevice));

    /**
     * GPU内にコピーしたBodies配列をsceneの中に格納する
     */
    sceneInitialize<<<1, 1>>>(d_scene, d_body);

    /// 1 threads per pixel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(ceil(static_cast<double>(h_image.getWidth()) / threadsPerBlock.x),
                       ceil(static_cast<double>(h_image.getHeight()) / threadsPerBlock.y));

    /// cuRand initialize ///
    checkCudaErrors(cudaMalloc((void**)&d_state, sizeof(curandState) * h_image.getWidth() * h_image.getHeight()));
    // seed value is const
    const int seed = 0;
    curandInitInRenderer<<<blocksPerGrid, threadsPerBlock>>>(d_scene, d_state, seed);

    writeToPixels<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, d_scene, samplesPerPixel, d_state);

    checkCudaErrors(cudaMemcpy(h_image.pixels, d_pixels, sizeof(Color) * h_image.getWidth() * h_image.getHeight(), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(d_body));
    checkCudaErrors(cudaFree(d_pixels));

    return h_image;
}

__device__ __forceinline__
void computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v) {
    if(fabs(w.x()) > 1e-3)
        u = Eigen::Vector3d(0, 1, 0).cross(w).normalized();
    else
        u = Eigen::Vector3d(1, 0, 0).cross(w).normalized();

    v = w.cross(u);
}

__device__ __forceinline__
void diffuseSample(const Eigen::Vector3d &normal, Ray &ray, const Eigen::Vector3d &incidentPoint, const double2 &rands) {
    const double phi = 2.0 * EIGEN_PI * rands.x;
    const double theta = acos(sqrt(rands.y));

    /// normalの方向をy軸とした正規直交基底を作る
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    const double _x = sin(theta) * cos(phi);
    const double _y = cos(theta);
    const double _z = sin(theta) * sin(phi);

    ray.dir = (_x * u + _y * normal + _z * v).normalized();
    ray.org = incidentPoint;
}

__device__ __forceinline__
void specularSample(const Eigen::Vector3d &normal, Ray &ray, const Eigen::Vector3d &incidentPoint) {
    ray.org = incidentPoint;
    ray.dir = (ray.dir - 2.0 * normal.dot(ray.dir) * normal).normalized();
}

__device__ __forceinline__
void refractSample(const Eigen::Vector3d &normal, Ray &ray, const Eigen::Vector3d &incidentPoint) {
    const double refidx1 = 1.0;
    const double refidx2 = 1.0;
    const double refidx = refidx1 / refidx2;
    const double dt = (-ray.dir).dot(normal);
    const double discriminant = 1.0 - refidx * refidx * (1.0 - dt * dt);

    ray.org = incidentPoint;
    if (discriminant > 0) ray.dir = refidx * (ray.dir + normal * dt) - normal * sqrt(discriminant);
    else ray.dir = Eigen::Vector3d::Zero();
}

__device__ __forceinline__
void pathTraceGPU(Color &radiance, const Scene *scene, Ray &ray, curandState &state, PATH_TRACE_FLAG &flag) {
    RayHit hit;

    if(!hitScene(scene, ray, hit)) {
        radiance = scene->backgroundColor;
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const Body hitBody = scene->bodies[hit.idx];
    const Material& hitBodyMaterial = hitBody.getMaterial();
    const Eigen::Vector3d incidentPoint = ray.org + hit.t * ray.dir;

    if(hitBody.getEmission() > 0.0) {
        /// ray hit a light source
        radiance = hitBody.getEmission() * radiance.cwiseProduct(hitBody.getMaterial().getColor());
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const double xi = generateRandom(state);

    if(xi < hitBodyMaterial.getKd()) {
        /// diffuse
        double2 rands = make_double2(generateRandom(state), generateRandom(state));
        diffuseSample(hit.normal, ray, incidentPoint, rands);
        radiance = radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    if(xi < hitBodyMaterial.getKd() + hitBodyMaterial.getKs()) {
        /// specular
        specularSample(hit.normal, ray, incidentPoint);
        radiance = radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    if(xi < hitBodyMaterial.getKd() + hitBodyMaterial.getKs() + hitBodyMaterial.getKt()) {
        /// specular
        refractSample(hit.normal, ray, incidentPoint);
        radiance = radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    radiance.setZero();
    flag = PATH_TRACE_TERMINATE;
}
