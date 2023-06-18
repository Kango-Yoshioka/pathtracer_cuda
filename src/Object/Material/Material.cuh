//
// Created by kango on 2023/01/06.
//

#ifndef LINKTEST_MATERIAL_CUH
#define LINKTEST_MATERIAL_CUH


#include "../../Image/Image.h"

enum MaterialPreset {
    M_DIFFUSE,
    M_SPECULAR,
    M_TRANSPARENT,
    M_ZERO
};

class Material {
private:
    Color color;
    double kd{};
    double ks{};
    double kt{};
public:
    Material(Color color, double kd, double ks, double kt);

    explicit Material(const MaterialPreset &materialPreset, const Color &color=Color::Zero(), const double &value=0.0);

    [[nodiscard]] __host__ __device__ const Color &getColor() const;

    [[nodiscard]] __host__ __device__ double getKd() const;

    [[nodiscard]] __host__ __device__ double getKs() const;

    [[nodiscard]] __host__ __device__ double getKt() const;
};


#endif//LINKTEST_MATERIAL_CUH
