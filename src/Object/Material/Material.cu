//
// Created by kango on 2023/01/06.
//

#include "Material.cuh"

Material::Material(const MaterialPreset &materialPreset, const Color &color, const double &value) {
    this->color = color;
    ks = 0; kd = 0; kt = 0;
    switch (materialPreset) {
        case M_DIFFUSE:
            kd = value;
            break;
        case M_SPECULAR:
            ks = value;
            break;
        case M_TRANSPARENT:
            kt = value;
            break;
        case M_ZERO:
            break;
    }
}

Material::Material(Color color, double kd, double ks, double kt) : color(std::move(color)), kd(kd), ks(ks), kt(kt) {
    if(kd + ks + kt > 1) {
        printf("Invalid value. kd+ks+kt>1\n");
        exit(EXIT_FAILURE);
    }
}

__host__ __device__ const Color &Material::getColor() const {
    return color;
}

__host__ __device__ double Material::getKd() const {
    return kd;
}

__host__ __device__ double Material::getKs() const {
    return ks;
}

__host__ __device__ double Material::getKt() const {
    return kt;
}