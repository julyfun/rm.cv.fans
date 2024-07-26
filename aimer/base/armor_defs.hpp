//
// Created by xinyang on 2020/7/19.
// 本文件是识别器和预测器的沟通桥梁

#ifndef AIMER_BASE_ARMOR_DEFS_HPP
#define AIMER_BASE_ARMOR_DEFS_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

#include "aimer/base/math/math.hpp"

namespace aimer {

// 内联？
enum class ArmorType { SMALL, BIG };

enum class EnemyType {
    OLD_SENTRY,
    HERO,
    ENGINEER,
    INFANTRY,
    BALANCE_INFANTRY,
    OUTPOST,
    CRYSTAL_BIG,
    CRYSTAL_SMALL
};

struct SampleArmor {
    constexpr SampleArmor(const aimer::ArmorType& type, const double& width, const double& height):
        type(type),
        width(width),
        height(height) {}
    aimer::ArmorType type;
    double width;
    double height;
};

constexpr double INF = 1e18;
// 用参数替代初始建模chr
constexpr int RESERVED_INT = 0;
constexpr double RESERVED_DOUBLE = 0.;
constexpr int MIN_ENEMY_NUMBER = 0;
constexpr int MAX_ENEMY_NUMBER = 8;

// 装甲板数据
constexpr aimer::SampleArmor BIG_ARMOR { aimer::ArmorType::BIG, 0.230, 0.127 };
constexpr aimer::SampleArmor SMALL_ARMOR { aimer::ArmorType::SMALL, 0.135, 0.125 };

} // namespace aimer

#endif /* TOS_ARMOR_DEFS_HPP */
