//
// Created by kango on 2023/01/04.
//

#include "Camera.cuh"

Camera::Camera(const Eigen::Vector3d &org, const Eigen::Vector3d &dir, const int &resolutionHeight, double aspectRatio, double verticalFoV, double focalLength, double focusDist, double fNumber, double sensitivity)
    : Ray(org, dir), focalLength(focalLength), focusDist(focusDist), fNumber(fNumber), sensitivity(sensitivity) {
    // 度数法からradianに変換
    const auto theta = verticalFoV * EIGEN_PI / 180.0;
    const double filmHeight = 2 * tan(theta / 2.0);
    film = Film(resolutionHeight, aspectRatio, filmHeight);

    // カメラとレンズの距離
    camToLensDist = focusDist * focalLength / (focusDist - focalLength);
    // レンズの場所を決定
    lensPos = org + dir.normalized() * camToLensDist;

    // カメラのローカル基底
    right = dir.cross(Eigen::Vector3d{0, 1, 0}).normalized();
    up = right.cross(dir).normalized();
}

__device__
void Camera::filmView(const unsigned int &p_x, const unsigned int &p_y, Ray &out_ray, double &weight, const double4 &rand) const {
    const auto pixelLocalPos = film.pixelLocalPosition(p_x, p_y, make_double2(rand.w, rand.x));

    // フィルムのピクセルのローカル座標をワールド座標に変換(カメラのorgを含む平面上に置く)
    const Eigen::Vector3d pixelWorldPos = org - film.filmSize.x() * right * (pixelLocalPos.x() - 0.5) - film.filmSize.y() * up * (0.5 - pixelLocalPos.y());
    // フィルムのピクセルとレンズの中心を結ぶベクトル
    const Eigen::Vector3d pixelToLens = (lensPos - pixelWorldPos).normalized();

    // レンズ上の位置サンプリング
    const double theta = 2.0 * EIGEN_PI * rand.y;
    const double r = focalLength / (2.0 * fNumber) * rand.z;
    const Eigen::Vector3d randPosOnLens = lensPos + r * (right * cos(theta) + up * sin(theta));

    // 集光点の計算(ボケずにくっきり映る点)
    const Eigen::Vector3d focalPos = lensPos + pixelToLens * focusDist / pixelToLens.dot(dir);

    out_ray.org = randPosOnLens;
    out_ray.dir = (focalPos - randPosOnLens).normalized();

    // weightの計算
    const auto pdf = 2.0 * EIGEN_PI / r;
    const auto cosine = fabs(out_ray.dir.dot(dir));
    weight = sensitivity * cosine * cosine * cosine * cosine / (pdf * camToLensDist * camToLensDist);
}