//
// Created by kango on 2022/12/30.
//
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Image.cuh"
#include "stb_image_write.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

Image::Image() = default;

__device__ __host__
Image::Image(const int &width, const int &height, const bool &gamma_correction) : width(width), height(height), gamma_correction(gamma_correction) {
    pixels = new Color[width * height];
    for(int i = 0; i < width * height; i++) {
        pixels[i] = Color::Zero();
    }
}

int Image::getWidth() const {
    return width;
}

int Image::getHeight() const {
    return height;
}

/**
 * 画像の中身を消去する（すべてのピクセルの輝度を0にする）
 */
[[maybe_unused]] void Image::clear() const {
    if(width == 0 || height == 0) {
        std::cerr << "width or height is an invalid value!!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < width * height; i++) {
        pixels[i] = Color::Zero();
    }
}

/**
 * 画像をbmp形式で出力する
 * ファイル名には拡張子は必要ありません
 * @param output_file_name_wo_extension
 */
void Image::generateBMP(const std::string &output_file_name_wo_extension) const {
    const std::string output_file_name = output_file_name_wo_extension + ".bmp";
    std::vector<unsigned char> imageData(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        auto r = std::clamp(int(pixels[i].x() * 255), 0, 255);
        auto g = std::clamp(int(pixels[i].y() * 255), 0, 255);
        auto b = std::clamp(int(pixels[i].z() * 255), 0, 255);
        int index = i * 3;
        imageData[index + 0] = r;
        imageData[index + 1] = g;
        imageData[index + 2] = b;
    }
    stbi_write_bmp(output_file_name.c_str(), width, height, 3, imageData.data());
}

/**
 * 画像をpng形式で出力する
 * ファイル名には拡張子は必要ありません
 * @param output_file_name_wo_extension
 */
void Image::generatePNG(const std::string &output_file_name_wo_extension) const {
    const std::string output_file_name = output_file_name_wo_extension + ".png";
    std::vector<unsigned char> imageData(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        /// gamma correction ///
        if(gamma_correction) {
            for(int j = 0; j < 3; j++) {
                if(pixels[i][j] <= 0.0031308) pixels[i][j] = pixels[i][j] * 12.92;
                else pixels[i][j] = pow(pixels[i][j], 1.0 / 2.4) * 1.055 - 0.055;
            }
        }

        auto r = std::clamp(int(pixels[i].x() * 255), 0, 255);
        auto g = std::clamp(int(pixels[i].y() * 255), 0, 255);
        auto b = std::clamp(int(pixels[i].z() * 255), 0, 255);
        int index = i * 3;
        imageData[index + 0] = r;
        imageData[index + 1] = g;
        imageData[index + 2] = b;
    }
    stbi_write_png(output_file_name.c_str(), width, height, 3, imageData.data(), width * 3);
}

/**
 * 画像をcsv形式で出力する
 * ファイル名には拡張子は必要ありません
 * @param output_file_name_wo_extension
 */
void Image::generateCSV(const std::string &output_file_name_wo_extension, const unsigned int &precision) const {
    std::ofstream ofs(output_file_name_wo_extension + ".csv");
    for(int i = 0; i < width * height; i++) {
        ofs << std::fixed << std::setprecision(precision) << pixels[i].x() << ","
            << std::fixed << std::setprecision(precision) << pixels[i].y() << ","
            << std::fixed << std::setprecision(precision) << pixels[i].z() << std::endl;
    }
}

/**
 * 画像を任意のファイル形式で出力する
 * ファイル名には拡張子は必要ありません
 * @param output_file_name_wo_extension
 * @param file_extension
 */
void Image::generateImageFile(const std::string &output_file_name_wo_extension, const FILE_EXTENSION &file_extension) const {
    switch (file_extension) {
        case BMP:
            generateBMP(output_file_name_wo_extension);
            break;
        case PNG:
            generatePNG(output_file_name_wo_extension);
            break;
        case CSV:
            generateCSV(output_file_name_wo_extension);
            break;
    }
}
