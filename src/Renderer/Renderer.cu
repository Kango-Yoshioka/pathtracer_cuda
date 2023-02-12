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

    if(p.x() >= scene->camera.resolution.x() || p.y() >= scene->camera.resolution.y()) return;

    const unsigned int pixelIdx = p.x() + (p.y() * scene->camera.resolution.x());
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

__device__ __host__
bool hitScene(const Scene *scene, const Ray &ray, RayHit &rayHit) {

    rayHit.t = FLT_MAX;
    rayHit.idx = -1;
    for (int i = 0; i < scene->bodiesSize; i++) {
        RayHit _rayHit;
        if (scene->bodies[i].hit(ray, _rayHit) && _rayHit.t < rayHit.t) {
            rayHit.t = _rayHit.t;
            rayHit.idx = i;
            rayHit.side = _rayHit.side;
            rayHit.normal = _rayHit.normal;
        }
    }
    return rayHit.idx != -1;
}

__host__
void generateImageWithCPU(const Scene &scene) {
    Image image(scene.camera.resolution.x(), scene.camera.resolution.y());
    auto *pixels = static_cast<Color*>(malloc(image.getWidth() * image.getHeight() * sizeof(Color)));

    std::random_device seedGen;
    std::default_random_engine engine(seedGen());
    std::uniform_real_distribution<> dist(0.0, 1.0);
#pragma omp parallel for
    for(int y = 0; y < scene.camera.resolution.y(); y++) {
        for(int x = 0; x < scene.camera.resolution.x(); x++) {
            const unsigned int pixelIdx = y * image.getWidth() + x;
            Ray initRay;
            Color radiance = Color().setZero();
            scene.camera.filmView(x, y, initRay);
            const int samplesPerPixel = 10000;
            for(int i = 0; i < samplesPerPixel; i++) {
                Ray in_ray = initRay; PATH_TRACE_FLAG flag; Color in_radiance;
                in_radiance.setOnes();
                do {
                    Ray out_ray;
                    Color out_radiance;
                    pathTraceCPU(in_radiance, out_radiance, &scene, in_ray, out_ray, flag, engine, dist);
                    in_ray = out_ray;
                    in_radiance = out_radiance;
                } while(flag);
                radiance += in_radiance;
            }
            pixels[pixelIdx] = radiance / static_cast<double>(samplesPerPixel);
        }
    }

    image.pixels = pixels;
    image.generatePNG("sampleCPU");
    image.generateCSV("sampleCPU");
    free(pixels);
}

__host__ void pathTraceCPU(const Color &in_radiance, Color &out_radiance, const Scene *scene, const Ray &in_ray, Ray &out_ray, PATH_TRACE_FLAG &flag, std::default_random_engine &engine, std::uniform_real_distribution<> &dist) {
    RayHit rayHit;
    const bool isHit = hitScene(scene, in_ray, rayHit);

    if(!isHit) {
        out_ray.org = Eigen::Vector3d{1, 2, 3};
        out_ray.dir = Eigen::Vector3d{1, 2, 3};
        out_radiance = in_radiance.cwiseProduct(Color(0.0, 0.0, 0.0));
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const Body hitBody = scene->bodies[rayHit.idx];
    const Material& hitBodyMaterial = hitBody.getMaterial();
    const Eigen::Vector3d incidentPoint = in_ray.org + rayHit.t * in_ray.dir;

    if(hitBody.getEmission() > 0.0) {
        /// in_ray hit a light source
        out_ray.org = incidentPoint;
        out_ray.dir = Eigen::Vector3d{3, 2, 1};
        out_radiance = in_radiance.cwiseProduct(hitBody.getEmission() * hitBody.getMaterial().getColor());
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const double xi = dist(engine);

    if(xi < hitBodyMaterial.getKd()) {
        /// diffuse
        double2 rands = make_double2(dist(engine), dist(engine));
        diffuseSample(rayHit.normal, out_ray, incidentPoint, rands);
        out_radiance = in_radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    out_ray.org = incidentPoint;
    out_ray.dir = Eigen::Vector3d{1, 2, 3};
    out_radiance = in_radiance.cwiseProduct(Color(0.0, 0.0, 0.0));
    flag = PATH_TRACE_TERMINATE;
}

__global__
void writeToPixels(Color *out_pixels, Scene *scene, unsigned int samplesPerPixel, curandState *states) {
    const Eigen::Vector2i p(
            (blockIdx.x * blockDim.x) + threadIdx.x,
            (blockIdx.y * blockDim.y) + threadIdx.y
    );
    if(p.x() >= scene->camera.resolution.x() || p.y() >= scene->camera.resolution.y()) return;
    const unsigned int pixelIdx = p.x() + (p.y() * scene->camera.resolution.x());

    curandState state = states[pixelIdx];
    Ray initRay;
    Color radiance = Color(0.0, 0.0, 0.0);

    scene->camera.filmView(p.x(), p.y(), initRay);
    for(int i = 0; i < samplesPerPixel; i++) {
        Ray in_ray = initRay;
        Color in_radiance;
        PATH_TRACE_FLAG pathTraceFlag;
        in_radiance.setOnes();
        do {
            Ray out_ray; Color out_radiance;
            pathTraceGPU(in_radiance, out_radiance, scene, in_ray, out_ray, state, pathTraceFlag);
            in_ray = out_ray;
            in_radiance = out_radiance;
        } while (pathTraceFlag);
        radiance += in_radiance;
    }

     out_pixels[pixelIdx] = radiance / static_cast<double>(samplesPerPixel);
}

__global__
void sceneInitialize(Scene *d_scene, Body *d_body) {
    d_scene->bodies = d_body;
}

__host__
void generateImageWithGPU(const Scene &scene, const unsigned int &samplesPerPixel) {
    Scene* d_scene;
    Body* d_body;
    Color* d_pixels;
    curandState* d_state;

    /// information of camera ///
    std::cout << "-- Camera --" << std::endl;
    std::cout << "Resolution\t(" << scene.camera.resolution.x() << ", " << scene.camera.resolution.y() << ")" << std::endl;
    std::cout << "------------" << std::endl;

    /// Scene initialize ///
    Image h_image(scene.camera.resolution.x(), scene.camera.resolution.y());
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
    curandInitInRenderer<<<blocksPerGrid, threadsPerBlock>>>(d_scene, d_state, static_cast<unsigned long>(time(nullptr)));

    cudaDeviceSynchronize();

    writeToPixels<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, d_scene, samplesPerPixel, d_state);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(h_image.pixels, d_pixels, sizeof(Color) * h_image.getWidth() * h_image.getHeight(), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(d_body));
    checkCudaErrors(cudaFree(d_pixels));

    const std::string fname = "sampleGPU";
    h_image.generatePNG(fname);
    h_image.generateCSV(fname);
}

__host__ __device__ __forceinline__
void computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v) {
    if(fabs(w.x()) > 1e-6)
        u = Eigen::Vector3d(0, 1, 0).cross(w).normalized();
    else
        u = Eigen::Vector3d(1, 0, 0).cross(w).normalized();

    v = w.cross(u);
}

__host__ __device__ __forceinline__
void diffuseSample(const Eigen::Vector3d &normal, Ray &out_ray, const Eigen::Vector3d &incidentPoint, const double2 &rands) {
    const double phi = 2.0 * EIGEN_PI * rands.x;
    const double theta = acos(sqrt(rands.y));

    /// normalの方向をy軸とした正規直交基底を作る
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    const double _x = sin(theta) * cos(phi);
    const double _y = cos(theta);
    const double _z = sin(theta) * sin(phi);

    out_ray.dir = (_x * u + _y * normal + _z * v).normalized();
    out_ray.org = incidentPoint;
}

__device__ __forceinline__
void specularSample(const Eigen::Vector3d &normal, const Ray &in_ray, Ray &out_ray, const Eigen::Vector3d &incidentPoint) {
    out_ray.org = incidentPoint;
    out_ray.dir = (in_ray.dir - 2.0 * normal.dot(in_ray.dir) * normal).normalized();
}

__device__ __forceinline__
void refractSample(const Eigen::Vector3d &normal, const Ray &in_ray, Ray &out_ray, const Eigen::Vector3d &incidentPoint) {
    const double refidx1 = 1.0;
    const double refidx2 = 1.0;
    const double refidx = refidx1 / refidx2;
    const double dt = (-in_ray.dir).dot(normal);
    const double discriminant = 1.0 - refidx * refidx * (1.0 - dt * dt);

    out_ray.org = incidentPoint;
    if (discriminant > 0) out_ray.dir = refidx * (in_ray.dir + normal * dt) - normal * sqrt(discriminant);
    else out_ray.dir = Eigen::Vector3d::Zero();
}

__device__ __forceinline__
void pathTraceGPU(const Color &in_radiance, Color &out_radiance, const Scene *scene, const Ray &in_ray, Ray &out_ray, curandState &state, PATH_TRACE_FLAG &flag) {
    RayHit rayHit;
    const auto isHit = hitScene(scene, in_ray, rayHit);

    if(!isHit) {
        out_radiance .setZero();
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const Body hitBody = scene->bodies[rayHit.idx];
    const Material& hitBodyMaterial = hitBody.getMaterial();
    const Eigen::Vector3d incidentPoint = in_ray.org + rayHit.t * in_ray.dir;

    if(hitBody.getEmission() > 0.0) {
        /// ray hit a light source
        out_radiance = hitBody.getEmission() * in_radiance.cwiseProduct(hitBody.getMaterial().getColor());
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const double xi = generateRandom(state);

    if(xi < hitBodyMaterial.getKd()) {
        /// diffuse
        double2 rands = make_double2(generateRandom(state), generateRandom(state));
        diffuseSample(rayHit.normal, out_ray, incidentPoint, rands);
        out_radiance = in_radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    if(xi < hitBodyMaterial.getKd() + hitBodyMaterial.getKs()) {
        /// specular
        specularSample(rayHit.normal, in_ray, out_ray, incidentPoint);
        out_radiance = in_radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    out_radiance.setZero();
    flag = PATH_TRACE_TERMINATE;
}

__device__ void pathTraceGPU2(const int &pixelIdx, const Color &in_radiance, Color &out_radiance, const Scene *scene, const Ray &in_ray, Ray &out_ray, curandState &state, PATH_TRACE_FLAG &flag) {
    RayHit rayHit;
    const auto isHit = hitScene(scene, in_ray, rayHit);

    if(!isHit) {
        out_radiance .setZero();
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const Body hitBody = scene->bodies[rayHit.idx];
    const Material& hitBodyMaterial = hitBody.getMaterial();
    const Eigen::Vector3d incidentPoint = in_ray.org + rayHit.t * in_ray.dir;

    if(hitBody.getEmission() > 0.0) {
        /// ray hit a light source
        out_radiance = hitBody.getEmission() * in_radiance.cwiseProduct(hitBody.getMaterial().getColor());
        flag = PATH_TRACE_TERMINATE;
        return;
    }

    const double xi = generateRandom(state);

    if(xi < hitBodyMaterial.getKd()) {
        /// diffuse
        const double2 rands = make_double2(generateRandom(state), generateRandom(state));
        diffuseSample(rayHit.normal, out_ray, incidentPoint, rands);
        out_radiance = in_radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    if(xi < hitBodyMaterial.getKd() + hitBodyMaterial.getKs()) {
        /// specular
        specularSample(rayHit.normal, in_ray, out_ray, incidentPoint);
        out_radiance = in_radiance.cwiseProduct(hitBodyMaterial.getColor());
        flag = PATH_TRACE_CONTINUE;
        return;
    }

    out_radiance.setZero();
    flag = PATH_TRACE_TERMINATE;
}
