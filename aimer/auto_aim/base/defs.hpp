#ifndef AIMER_AUTO_AIM_BASE_DEFS_HPP
#define AIMER_AUTO_AIM_BASE_DEFS_HPP

#include <Eigen/Core>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

#include "aimer/base/armor_defs.hpp"

namespace aimer {

const std::vector<cv::Point3d> PW_SMALL = { // 灯条坐标，单位：m
    { -0.068, 0.0275, 0. },
    { -0.068, -0.0275, 0. },
    { 0.068, -0.0275, 0. },
    { 0.068, 0.0275, 0. }
};
const std::vector<cv::Point3d> PW_BIG = { // 灯条坐标，单位：m
    { -0.115, 0.0265, 0. },
    { -0.115, -0.0265, 0. },
    { 0.115, -0.0265, 0. },
    { 0.115, 0.0265, 0. }
};
// ENEMY 标号的最大值

/*
哨兵0（BIG: 1)
英雄1   (BIG : 4 有陀螺)
四板步兵&工程车2345 (SMALL 4 有陀螺)
平衡步兵345 (BIG 1 (2)，不构建反陀螺，当作单板子打好了)
前哨站6 (SMALL 1)
水晶大7 (BIG 1)
水晶小8 (SMALL 1)
*/

struct DetectedArmor {
    cv::Point2f pts[4];
    int color;
    int number;
    float conf;
    float conf_class;
};

struct DetectionResult {
    cv::Mat img;
    Eigen::Quaternionf q;
    double timestamp;
    std::vector<aimer::DetectedArmor> armors;
};

// 神经网络传来的 armor 计算得到的装甲板数据
struct ArmorInfo {
    int frame; // 所属帧
    aimer::DetectedArmor detected;
    aimer::SampleArmor sample;
    std::array<cv::Point2f, 4> pus; // pts undistorted 畸变矫正后的坐标，用于重投影后比较
    // 用在三分法求 yaw 里面了
    double orientation_pitch_under_rule;
    Eigen::Vector3d pos; // 陀螺仪 xyz 坐标
    Eigen::Quaterniond rotation_q;
    aimer::math::YpdCoord orientation_yp;

    bool valid() const {
        if (!(this->detected.number >= 0 && this->detected.number <= aimer::MAX_ENEMY_NUMBER)) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (aimer::math::is_nan_or_inf(this->detected.pts[i].x)
                || aimer::math::is_nan_or_inf(this->detected.pts[i].y))
            {
                return false;
            }
        }
        for (int i = 0; i < 3; ++i) {
            if (aimer::math::is_nan_or_inf(this->pos(i, 0))) {
                return false;
            }
        }
        return true;
    }
    float area() const {
        return aimer::math::get_area(this->detected.pts);
    }
    cv::Point2f center() const {
        return cv::Point2f { (this->detected.pts[0].x + this->detected.pts[1].x
                              + this->detected.pts[2].x + this->detected.pts[3].x)
                                 / 4.f,
                             (this->detected.pts[0].y + this->detected.pts[1].y
                              + this->detected.pts[2].y + this->detected.pts[3].y)
                                 / 4.f };
    }
};

struct ArmorData {
    // 生成一个装甲板实例数据
    ArmorData(const int& id, const int& color, const aimer::ArmorInfo& info):
        id(id),
        color(color),
        info(info) {}
    bool is_hit() const {
        return this->color != this->info.detected.color;
    }
    int id;
    int color;
    aimer::ArmorInfo info;
};

} // namespace aimer

#endif /* AIMER_AUTO_AIM_BASE_DEFS_HPP */
