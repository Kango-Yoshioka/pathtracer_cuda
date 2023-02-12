//
// Created by kango on 2022/12/30.
//

#ifndef CUDATEST_IMAGE_CUH
#define CUDATEST_IMAGE_CUH

#include "Eigen/Dense"

using Color = Eigen::Vector3d;

enum FILE_EXTENSION {
    BMP, PNG, CSV
};

class Image {
private:
    int width{}, height{};
    bool gamma_correction{};
public:
    Color *pixels{};

    Image();

    __device__ __host__
    Image(const int &width, const int &height, const bool &gamma_correction=true);

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[maybe_unused]] void clear() const;

    void generateBMP(const std::string &output_file_name_wo_extension) const;

    void generateCSV(const std::string &output_file_name_wo_extension, const unsigned int &precision=8) const;

    void generatePNG(const std::string &output_file_name_wo_extension) const;

    void generateImageFile(const std::string &output_file_name_wo_extension, const FILE_EXTENSION &file_extension) const;
};


#endif //CUDATEST_IMAGE_CUH
