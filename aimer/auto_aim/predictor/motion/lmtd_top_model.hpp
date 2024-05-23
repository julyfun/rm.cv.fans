#ifndef AIMER_AUTO_AIM_PREDICTOR_MOTION_LMTD_TOP_MODEL_HPP
#define AIMER_AUTO_AIM_PREDICTOR_MOTION_LMTD_TOP_MODEL_HPP
#include <Eigen/Core>
#include <ceres/ceres.h>

#include "aimer/auto_aim/predictor/enemy/enemy_state.hpp"
#include "aimer/auto_aim/predictor/motion/top4_model.hpp"
#include "aimer/auto_aim/predictor/motion/top_model.hpp"
#include "aimer/base/math/filter/adaptive_ekf.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"

// ehh lmtd must be very happy about this name.
namespace aimer::lmtd {

constexpr int N_X = 9;
constexpr int N_Z = 4;
// EKF
// xa = x_armor, xc = x_robot_center
// state: xc, v_xc, yc, v_yc, za, v_za, orient, v_yaw, r,
// index: 0,  1,    2,  3,    4,  5,    6,      7,     8,
// - yaw 需要在切换时 set?
// - 对于只有单独半径的装甲板， 不更新多余的 r 和 z
// measurement: xa, ya, za, yaw
// measurement2: yaw（相机）, pitch, distance, yaw(orient)
// f - Process function

using Ekf = aimer::AdaptiveEkf<N_X, N_Z>;
using State = Eigen::Matrix<double, N_X, 1>;
using Observation = Eigen::Matrix<double, N_Z, 1>;

// 这只是用于绘制图像的一个试验品结构体
struct PredictedArmor {
    Eigen::Vector3d pos;
    double orientation_yaw;
    double orientation_pitch;
    aimer::ArmorType type;
};

// 轮式轨迹滤波器
struct ArmorFilter: public aimer::PositionPredictorInterface {
private:
    State state;
    double t;

public:
    ArmorFilter(const State& state, const double& t): state(state), t(t) {}
    State predict(const double& t) const;
    Eigen::Vector3d predict_pos(const double& t) const override;
    Eigen::Vector3d predict_v(const double& t) const override;
};

struct AimAndState {
    aimer::AimInfo aim;
    State state;
};

struct TopModel {
private:
    double predict_t;
    double update_t;
    Ekf ekf;
    double another_radius;
    double dz;
    // 正在追踪的装甲板的序号
    // 序号的计算依赖 aimer::LightThread
    int tracked_armor_id;
    // 计算目标相关
    int top_level;

public:
    // [functions]
    void init(aimer::CoordConverter* const converter, aimer::EnemyState* const enemy_state);
    void update(aimer::CoordConverter* const converter, aimer::EnemyState* const enemy_state);
    bool credit(aimer::CoordConverter* const converter) const;
    std::vector<PredictedArmor> predict_armors(
        const double& t,
        aimer::CoordConverter* const converter,
        aimer::EnemyState* const state
    ) const;
    void draw_armors(
        cv::Mat& img,
        const double& t,
        aimer::CoordConverter* const converter,
        aimer::EnemyState* const state
    ) const;

    int get_top_level() const;

    std::vector<ArmorFilter> get_armor_filters(aimer::EnemyState* const enemy_state) const;

    aimer::AimInfo
    get_aim(aimer::CoordConverter* const converter, aimer::EnemyState* const enemy_state) const;

private:
    // 这个函数用到了 this->tracked_armor_id
    std::tuple<Eigen::Matrix<double, N_Z, 1>, std::array<double, N_Z>, int>
    get_observation_and_r_and_id(
        aimer::CoordConverter* const converter,
        aimer::EnemyState* const enemy_state
    ) const;

    void update_top_level();
};
} // namespace aimer::lmtd

#endif
