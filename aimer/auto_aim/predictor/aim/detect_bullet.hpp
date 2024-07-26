#ifndef AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_HPP
#define AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_HPP

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "do_reproj.hpp"

namespace aimer::aim {
class ImageBullet {
  public:
    cv::Point2f center;
    float radius;

    ImageBullet() {}
    ImageBullet(const cv::Point2f& _center, const float& _radius) :
        center(_center),
        radius(_radius) {}
};

class DoFrameDifference {
    cv::Mat src1, src2;
    cv::Mat kernel1;

    double tme;

  public:
    DoFrameDifference() {
        kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
    }

    cv::Mat get_diff(
        const cv::Mat& s1,
        const cv::Mat& s2,
        const cv::Mat& ref,
        const cv::Mat& lst_fr_bullets);

    ~DoFrameDifference() {
        // std::cerr << "DoFrameDifference - get_diff : " << tme << " s\n";
    }
};

class DetectBullet {
    cv::Mat lst_frame, cur_frame;
    Eigen::Quaterniond lst_fr_q, cur_fr_q;  // 上一帧和当前帧的相机姿态
    cv::Mat cur_hsv;  // 当前帧的 HSV
    cv::Mat lst_hsv;  // 上一帧的 HSV
    cv::Mat lst_msk;  // 上一次识别为子弹的部分
    cv::Mat kernel1, kernel2;  // 形态学运算 kernel1

    DoReproj do_reproj;  // 用来做重投影
    DoFrameDifference do_diff;  // 做帧差

    std::vector<std::vector<cv::Point>> contours;

    std::vector<std::vector<uint32_t>> sort_pts;  // 用于快速排序的 vector

    void get_possible();

    void sort_points(std::vector<cv::Point>& vec);

    bool test_is_bullet(std::vector<cv::Point> contour);
    void get_bullets();

    double tme_get_brightest, tme_sort_points, tme_find_contours, tme_min_area_rect,
        tme_get_possible, tme_total;  // 记录使用时间

  public:
    cv::Mat tmp_output;

    DetectBullet();
    DetectBullet(const DoReproj& do_reproj);

    void init(const DoReproj& do_reproj);

    std::vector<ImageBullet> bullets;

    cv::Mat print_bullets();
    std::vector<ImageBullet>
    process_new_frame(const cv::Mat& new_frame, const Eigen::Quaterniond& q);

    ~DetectBullet() {
        // std::cerr << "DetectBullet - get_brightest : " << tme_get_brightest
        //           << " s \n";
        // std::cerr << "DetectBullet - sort_points : " << tme_sort_points << " s
        // \n"; std::cerr << "DetectBullet - findContours : " << tme_find_contours
        //           << " s \n";
        // std::cerr << "DetectBullet - minAreaRect : " << tme_min_area_rect
        //           << " s \n";
        // std::cerr << "DetectBullet - get_possible : " << tme_get_possible
        //           << " s \n";
        // std::cerr << "DetectBullet - Total time : " << tme_total << " s \n";
    }
};
}  // namespace aimer::aim

#endif /* AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_HPP */
