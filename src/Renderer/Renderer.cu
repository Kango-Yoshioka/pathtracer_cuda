//
// Created by kango on 2023/01/02.
//

#include "Renderer.cuh"
#include "helper_cuda.h"
#include <iostream>

/// Random number Generator Functions
__global__ void curandInitInRenderer(const Scene *scene, curandState* state, unsigned long seed) {
    const Eigen::Vector2i p(
            blockIdx.x, blockIdx.y
    );

    if(p.x() >= scene->camera.film.resolution.x() || p.y() >= scene->camera.film.resolution.y()) return;

    const unsigned int pixelIdx = p.x() + (p.y() * scene->camera.film.resolution.x());
    curand_init(seed, blockDim.x * pixelIdx + threadIdx.x, 0, &state[blockDim.x * pixelIdx + threadIdx.x]);
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
void writeToPixels(Color *out_pixelBuffer, Scene *scene, unsigned int samplesPerPixel, curandState *states) {
    const Eigen::Vector2i p(
            blockIdx.x, blockIdx.y
    );

    if(p.x() >= scene->camera.film.resolution.x() || p.y() >= scene->camera.film.resolution.y()) return;

    const unsigned int pixelIdx = p.x() + (p.y() * scene->camera.film.resolution.x());

    curandState state = states[blockDim.x * pixelIdx + threadIdx.x];
    Color accumulate_radiance = Color::Zero();
    const int samplesPerThreads = samplesPerPixel / blockDim.x;

    for(int i = 0; i < samplesPerThreads; i++) {
        const double4 rand = make_double4(generateRandom(state), generateRandom(state), generateRandom(state), generateRandom(state));
        Ray ray; double weight;
        Color radiance = Color::Ones();
        scene->camera.filmView(p.x(), p.y(), ray, weight, rand);
        while(true) {
            RayHit hit;
            if(!hitScene(scene, ray, hit)) {
                radiance = scene->backgroundColor;
                break;
            }

            const Body hitBody = scene->bodies[hit.idx];
            const Material& hitBodyMaterial = hitBody.getMaterial();

            if(hitBody.getEmission() > 0.0) {
                /// ray hit a light source
                radiance = hitBody.getEmission() * radiance.cwiseProduct(hitBody.getMaterial().getColor());
                break;
            }

            const double xi = generateRandom(state);
            const Eigen::Vector3d incidentPoint = ray.org + hit.t * ray.dir;

            if(xi < hitBodyMaterial.getKd()) {
                /// diffuse
                double2 rands = make_double2(generateRandom(state), generateRandom(state));
                diffuseSample(hit.normal, ray, incidentPoint, rands);
                radiance = radiance.cwiseProduct(hitBodyMaterial.getColor());
            } else if(xi < hitBodyMaterial.getKd() + hitBodyMaterial.getKs()) {
                /// specular
                specularSample(hit.normal, ray, incidentPoint);
                radiance = radiance.cwiseProduct(hitBodyMaterial.getColor());
            } else if(xi < hitBodyMaterial.getKd() + hitBodyMaterial.getKs() + hitBodyMaterial.getKt()) {
                /// specular
                refractSample(hit.normal, ray, incidentPoint);
                radiance = radiance.cwiseProduct(hitBodyMaterial.getColor());
            } else {
                radiance.setZero();
                break;
            }
        }
        accumulate_radiance += weight * radiance;
    }

    out_pixelBuffer[blockDim.x * pixelIdx + threadIdx.x] = accumulate_radiance / static_cast<double>(samplesPerPixel);
}

void genKeys(const int &N, const int &group_size, thrust::device_vector<int> &d_key) {
    thrust::counting_iterator<int> iter(0);

    auto keys = thrust::make_transform_iterator(iter, [group_size] __host__ __device__(int i) {
        return i / group_size;
    });

    d_key = thrust::device_vector<int>(N);
    thrust::copy(keys, keys + N, d_key.begin());
}

void reductionBuffer(thrust::device_vector<Color> &in_pixelBuffer, Color *out_pixels, const int width, const int height, const int threadsPerPixel) {
    const int N = width * height * threadsPerPixel;
    thrust::device_vector<int> in_key, out_key;
    thrust::device_vector<Color> d_pixels(width * height);
    genKeys(N, threadsPerPixel, in_key);
    out_key = thrust::device_vector<int>(width * height);

    thrust::reduce_by_key(
            in_key.begin(), in_key.end(), in_pixelBuffer.begin(),
            out_key.begin(), d_pixels.begin(), thrust::equal_to<int>(), thrust::plus<Color>());

    checkCudaErrors(cudaMemcpy(out_pixels, thrust::raw_pointer_cast(d_pixels.data()), sizeof(Color) * width * height, cudaMemcpyDeviceToHost));
}

__global__
void sceneInitialize(Scene *d_scene, Body *d_body) {
    d_scene->bodies = d_body;
}

__host__
Image generateImageWithGPU(const Scene &scene, const unsigned int &samplesPerPixel) {
    Scene* d_scene;
    Body* d_body;
    thrust::device_vector<Color> d_pixelBuffer;
    curandState* d_state;

    /// information of camera ///
    std::cout << "-- Camera --" << std::endl;
    std::cout << "Resolution\t(" << scene.camera.film.resolution.x() << ", " << scene.camera.film.resolution.y() << ")" << std::endl;
    std::cout << "------------" << std::endl;

    /// Scene initialize ///
    Image h_image(scene.camera.film.resolution.x(), scene.camera.film.resolution.y());
    std::cout << "Pixel size:\t" << h_image.getWidth() * h_image.getHeight() << std::endl;
    std::cout << "Number of samples:\t" << samplesPerPixel << std::endl;

    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
    /**
     * 構造体内に配列がある場合、cudaMemcpyをしても中身まではコピーされず、エラーとなるので、
     * 構造体内の配列は別途でGPUに送る必要あり。
     */
    // 1pixelあたりのthread数
    const int threadsPerPixel = 128;
    checkCudaErrors(cudaMalloc((void**)&d_body, sizeof(Body) * scene.bodiesSize));
    d_pixelBuffer = thrust::device_vector<Color>(h_image.getWidth() * h_image.getHeight() * threadsPerPixel);

    checkCudaErrors(cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_body, scene.bodies, sizeof(Body) * scene.bodiesSize, cudaMemcpyHostToDevice));

    /**
     * GPU内にコピーしたBodies配列をsceneの中に格納する
     */
    sceneInitialize<<<1, 1>>>(d_scene, d_body);

    /// 1 blocks per pixel
    dim3 threadsPerBlock(threadsPerPixel);
    dim3 blocksPerGrid(h_image.getWidth(), h_image.getHeight());

    /// cuRand initialize ///
    checkCudaErrors(cudaMalloc((void**)&d_state, sizeof(curandState) * h_image.getWidth() * h_image.getHeight() * threadsPerPixel));
    // seed value is const
    const int seed = 0;
    curandInitInRenderer<<<blocksPerGrid, threadsPerBlock>>>(d_scene, d_state, seed);

    writeToPixels<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_pixelBuffer.data()), d_scene, samplesPerPixel, d_state);

    reductionBuffer(d_pixelBuffer, h_image.pixels, h_image.getWidth(), h_image.getHeight(), threadsPerPixel);

    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(d_body));

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

