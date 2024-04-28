#ifndef AIMER_AUTO_AIM_PREDICTOR_MOTION_TOP_MODEL_HPP
#define AIMER_AUTO_AIM_PREDICTOR_MOTION_TOP_MODEL_HPP

#include <fmt/format.h>

#include <Eigen/Dense>
#include <iostream> // debug 用
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <utility> // pair
#include <vector>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/predictor/enemy/enemy_state.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/filter/filter.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"
#include "base/param/parameter.hpp"

namespace aimer::top {

enum class TopDirection {
    // 逆时针，左到右
    CCW,
    // 顺时针，右到左
    CW
};

// 旋转预测输出
// TopFutureArmor, with member t
struct TopArmor {
    TopArmor(
        const Eigen::Vector3d& pos,
        const double& t,
        const double& zn_to_v,
        const double& length,
        const double& z_plus
    ):
        pos(pos),
        t(t),
        zn_to_v(zn_to_v),
        length(length),
        z_plus(z_plus) {}
    Eigen::Vector3d pos;
    double t;
    // 陀螺仪 xy 平面上，相机 z 轴反方向旋转到装甲板法向量（方向向外）的角度
    double zn_to_v;
    // 半径
    double length;
    double z_plus;
};
// 本滤波器在非连续场所使用，且无法在第一次使用时 set_pos。
// 前置数据滤波生成器为预测二维的 AngleFilter
// AngleFilter 存在少量非正态误差，生成的估计角度
// 的误差是一个波形
using CenterFilter = PositionFilter<2, 1>;

double get_pts_cost(
    const std::vector<cv::Point2f>& cv_refs,
    const std::vector<cv::Point2f>& cv_pts,
    const double& inclined
);

std::vector<Eigen::Vector3d> radial_armor_corners(
    const Eigen::Vector3d& pos,
    const aimer::ArmorType& type,
    const double& pitch,
    const double& z_to_v,
    aimer::CoordConverter* const converter
);

std::vector<cv::Point2f> radial_armor_pts(
    const Eigen::Vector3d& pos,
    const aimer::ArmorType& type,
    const double& pitch,
    const double& z_to_v,
    aimer::CoordConverter* const converter
);

std::vector<std::vector<cv::Point2f>> radial_double_pts(
    const std::vector<aimer::ArmorData>& armors,
    const double& z_to_l,
    aimer::CoordConverter* const converter
);

// 单板拟合所需数据
struct SingleData {
    SingleData(
        const aimer::ArmorData& d,
        const double& z_to_v_exp, // 也就是 inclined
        aimer::CoordConverter* const converter
    ):
        d(d),
        z_to_v_exp(z_to_v_exp),
        converter(converter) {}
    const aimer::ArmorData d;
    const double z_to_v_exp;
    aimer::CoordConverter* const converter;
};

// 单板拟合代价类
class SingleCost {
public:
    explicit SingleCost(const top::SingleData& data): data(data) {}
    double operator()(const double& x);

private:
    const top::SingleData data;
};

// class SingleCostFunctor {
//  public:
//   SingleCostFunctor(const top::SingleData& data) : data(data) {}

//   template <typename T>
//   bool operator()(const T* const x, T* residual) const {
//     std::vector<cv::Point2f> pts = top::radial_armor_pts(
//         this->data.d.info.pos, this->data.d.info.sample.type,
//         this->data.d.info.pitch, double(x[0]), this->data.converter);
//     residual[0] =
//         top::get_pts_cost(pts,
//                           std::vector<cv::Point2f>(this->data.d.info.pus,
//                                                    this->data.d.info.pus +
//                                                    4),
//                           this->data.z_to_v_exp);
//     return true;
//   }

//  private:
//   top::SingleData data;
// };

struct DoubleData {
    DoubleData(
        const std::vector<aimer::ArmorData>& armors,
        const double& z_to_l_exp,
        aimer::CoordConverter* const converter
    ):
        armors(armors),
        z_to_l_exp(z_to_l_exp),
        converter(converter) {}
    const std::vector<aimer::ArmorData> armors;
    const double z_to_l_exp;
    aimer::CoordConverter* const converter;
};

class DoubleCost {
public:
    explicit DoubleCost(const top::DoubleData& data): data(data) {}
    double operator()(const double& x);

private:
    const top::DoubleData data;
};

// 拟合得到角度
// 为了方便，其中 z_to_v 开头的参数介于 0 ~ 2pi
double fit_single_z_to_v(
    const aimer::ArmorData& armor,
    const double& z_to_v_exp,
    const double& z_to_v_min,
    const double& z_to_v_max,
    aimer::CoordConverter* const converter
);

// 拟合得到角度。数据，限制，数据库
double fit_double_z_to_l(
    const std::vector<aimer::ArmorData>& armors,
    const double& z_to_l_exp,
    const double& z_to_l_min,
    const double& z_to_l_max,
    aimer::CoordConverter* const converter
);

// 不应该用模板限制类的行为，决策机应该允许使用较多 if

/**
 * @brief 获取打击陀螺模型所需的控制指令
 * 
 * @param prediction_results 此时下达转动指令，control 时间点发射的子弹的击中时间
 * @param hit_results 此时下达开火指令，这一指令对应发射子弹的击中时间
 * @return aimer::AimInfo 
 */
aimer::AimInfo get_top_limit_aim(
    const top::CenterFilter& center_filter,
    const std::vector<TopArmor>& prediction_results,
    const std::vector<TopArmor>& hit_results,
    const double& top_w,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state
);

struct TopDrawData {
    bool active;
    double w;
    Eigen::Vector3d center;
    std::vector<TopArmor> armors;
};

// 绘制全车装甲板
void top_draw_aim(
    cv::Mat& img,
    const top::CenterFilter& center_filter,
    const std::vector<TopArmor>& top_results,
    bool top_active,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state
);

// 反向延长得到中心
Eigen::Vector3d prolonged_center(
    const Eigen::Vector3d& armor_pos,
    const double& radius,
    const Eigen::Vector2d& radius_norm,
    const double& z_plus
);

class SimpleRotateCounter {
public:
    SimpleRotateCounter(
        const int& armor_cnt,
        const double& jump_angle,
        const double& min_active_rotate
    );

    int get_active_rotate() const;
    int get_rotate() const;
    double get_w() const;
    double predict_theta_v(const double& t) const;

    bool w_active() const;
    bool w_inactive() const;
    bool top_active() const;

    void update_theta_v(const double& theta_v, const double& t, const double& R);
    // 在 update_rotate 之前并不知道 predict_v 是否真的是 l
    void update_rotate(const aimer::ArmorData& d, const double& zn_to_v, const double& t);

private:
    const int armor_cnt;
    const double jump_angle;
    const int min_active_rotate;
    top::TopDirection id_direction = top::TopDirection::CCW;
    aimer::AngleFilter<1, 2> theta_s_filter { 2. * M_PI,
                                              { 0.01, 10., 10. },
                                              aimer::RESERVED_DOUBLE };
    int rotate = 0; // l 板到 super 板需要旋转多少个 120 / 180 度
    int active_rotate = 0; // 活跃旋转次数
    int last_id = -1;
    int last_seg = 0;
};

class AngleReserver {
public:
    void init(const double& angle);
    void update(const double& angle);
    double get_early();
    double get_late();

private:
    double early;
    double late;
};

// 定中心简单陀螺装甲板朝向角符号修正机
// 对于中心静止的陀螺目标，可以利用对方装甲板在视野中的 yaw 规律判断
// 装甲板朝向角的符号

struct OrientationSignFixerConstructor {
    OrientationSignFixerConstructor(
        const double& reset_time,
        const double& sampling_time,
        const double& reserving_range
    ):
        reset_time(reset_time),
        sampling_time(sampling_time),
        reserving_range(reserving_range) {}
    const double& reset_time;
    const double& sampling_time;
    const double& reserving_range;
};

class OrientationSignFixer {
public:
    explicit OrientationSignFixer(const OrientationSignFixerConstructor& cons);

    // 用上一周期的 yaw 修复 angle 的符号
    double fixed(const double& angle, const double& yaw) const;
    void update(const double& yaw, const double& t);

private:
    const double reset_time; // 取样时间上限
    const double sampling_time; // 取样时间下限
    const double reserving_range;
    AngleReserver reserver;
    double update_t = -86400.;
    double init_t = -86400.;
    double credit_min_yaw = 0.;
    double credit_max_yaw = 0.;
};

class SimpleTopModel {
public:
    SimpleTopModel(
        aimer::CoordConverter* const converter,
        aimer::EnemyState* const state,
        const int& armor_cnt,
        const double& jump_angle,
        const int& min_active_rotate,
        const double& credit_time,
        const OrientationSignFixerConstructor& cons
    );

    ~SimpleTopModel() = default;

    void update(const bool& enable_sign_fixer);
    bool active() const;
    std::vector<TopArmor> predict_armors(const double&) const;
    aimer::AimInfo get_limit_aim() const;
    void draw_aim(cv::Mat&) const;

private:
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    const int armor_cnt;
    aimer::CreditClock credit_clock;
    // 不能永久确定位置的 center 不可使用 1 维度更新
    // 还是用 3更新 - 2预测 组合 或者 3更新 - 1预测
    top::CenterFilter center_filter;
    top::SimpleRotateCounter counter;
    top::OrientationSignFixer orientation_sign_fixer;
    int tracking_id = -1;
};

} // namespace aimer::top

#endif /* AIMER_AUTO_AIM_PREDICTOR_MOTION_TOP_MODEL_HPP */

#endif /* AIMER_AUTO_AIM_TOP_MODEL_HPP */
