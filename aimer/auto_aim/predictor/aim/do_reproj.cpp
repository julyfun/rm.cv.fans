#ifndef __DO_REPROJ_CPP__
#define __DO_REPROJ_CPP__

#include "do_reproj.hpp"

#include <opencv2/core/eigen.hpp>

namespace aimer::aim {

DoReproj::DoReproj() {}

void DoReproj::init(const cv::Mat& cam, const cv::Mat& imu) {
    this->cam = Eigen::Matrix4d();
    Eigen::Matrix<double, 3, 4> mat;
    cv::cv2eigen(cam, mat);
    // for inversibility (théorème de factorisation)
    this->cam.block<3, 4>(0, 0) = mat;
    this->cam(3, 3) = 1;

    cv::cv2eigen(imu, this->imu);
}

DoReproj::DoReproj(const cv::Mat& cam, const cv::Mat& imu) {
    this->init(cam, imu);
}

Eigen::Matrix4d DoReproj::from_q_get_trans_mat(const DoReproj::Quat& q) {
    Eigen::Matrix4d res = Eigen::Matrix4d::Zero(4, 4);
    res.block<3, 3>(0, 0) = this->imu * q.matrix().inverse();
    res(3, 3) = 1;
    return res;
}

Eigen::Matrix3d DoReproj::get_fr_trans_mat(const DoReproj::Quat& q1, const DoReproj::Quat& q2) {
    Eigen::Matrix4d mat = this->cam * this->from_q_get_trans_mat(q2)
        * (this->cam * this->from_q_get_trans_mat(q1)).inverse();
    return mat.block<3, 3>(0, 0);
}

cv::Mat DoReproj::reproj(const cv::Mat& src, const DoReproj::Quat& q1, const DoReproj::Quat& q2) {
    // get transform matrix
    Eigen::Matrix3d mat = this->get_fr_trans_mat(q1, q2);
    cv::Mat cv_mat;
    cv::eigen2cv(mat, cv_mat);

    cv::Mat res;
    cv::warpPerspective(src, res, cv_mat, src.size());
    return res.clone();
}
}  // namespace aimer::aim
#endif
