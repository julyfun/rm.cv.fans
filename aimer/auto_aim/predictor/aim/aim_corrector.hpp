// 辅助计算发射误差的模型。
// 推导见 ../docs/latency.md。
// 此校正方法假设瞄准(aim)
// 计算无误，发射误差来源仅为机械结构、电控或视觉补偿参数错误。

#ifndef AIMER_AUTO_AIM_PREDICTOR_AIM_AIM_CORRECTOR_HPP
#define AIMER_AUTO_AIM_PREDICTOR_AIM_AIM_CORRECTOR_HPP

#include <list>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/predictor/aim/detect_bullet.hpp"
#include "aimer/base/math/filter/filter.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"

namespace aimer::aim {

/**
 * @brief 记录瞄准命令的 id，原始图像时间，瞄准参数，校正参数
 */
struct IdTLatencyAimCorrection {
    enum { INVALID_ID = -1 };
    int id;
    double img_t; // scheduled (implicit) = t + latency
    double img_to_predict_latency;
    aimer::AimInfo aim;
    Eigen::Vector2d correction;
    static auto invalid() -> IdTLatencyAimCorrection {
        return IdTLatencyAimCorrection { IdTLatencyAimCorrection::INVALID_ID,
                                         0.,
                                         0.,
                                         aimer::AimInfo::idle(),
                                         Eigen::Vector2d(0., 0.) };
    }
};

/**
 * @brief 以 IdTAimCorrection 形式记录最近的瞄准(aim) 命令
 */
class AimHistory {
public:
    AimHistory(const std::size_t& max_sz);

    // 非常神奇，当一个 id 被激活时，我们查询该 id 对应的 cmd 的估计发射时间 t
    // 然后寻找 t 最近的 aimer::AimInfo，其中的 target_pos 可以来复现弹道
    // id 命令发出的子弹对应的 aim 并不是 id 的
    auto add_aim(const aim::IdTLatencyAimCorrection& aim) -> void;
    auto get_id_cnt() const -> int;

    // log 复杂度寻找 t 后面最接近的 aim
    // 返回的 id 若是 0，表示未找到
    auto find_by_img_t(const double& img_t) const -> aim::IdTLatencyAimCorrection;
    auto find_by_id(const int& id) const -> aim::IdTLatencyAimCorrection;

private:
    const std::size_t max_sz;
    int id_cnt = 0;
    // 没有声明新签名，最近具名空间为 AimHistory，所以可直接 IdTAim。
    std::deque<aim::IdTLatencyAimCorrection> aims;
};

/**
 * @class ProjectileSimulator
 * @brief 存储一个子弹的发射信息并随时模拟其位置
 */
struct HitPos {
    bool hit;
    Eigen::Vector3d pos;
};

struct HitCircle {
    bool hit;
    aimer::math::CircleF circle;
};

struct CaughtCost {
    bool caught;
    double cost;
};

class ProjectileSimulator {
public:
    ProjectileSimulator(
        aimer::CoordConverter* const converter,
        const aim::IdTLatencyAimCorrection& aim,
        const double& fire_t
    ):
        converter { converter },
        aim { aim },
        fire_t { fire_t } {}

    auto get_aim_ref() const -> const aim::IdTLatencyAimCorrection&;
    auto get_fire_t() const -> double;

    // 声明了新的签名 get_pos，最近具名 scope 为 get_pos，需加 Proj.. 前缀。
    // 保留 via t 接口是因为可以考虑二分 t 获取最优半径。

    auto get_pos_by_t(const double& t) const -> aim::HitPos;
    auto get_pos() const -> aim::HitPos;

    auto get_circle_by_t(const double& t) const -> aim::HitCircle;
    auto get_circle() const -> aim::HitCircle;

    auto catch_circle(const aimer::math::CircleF& circle) const -> aim::CaughtCost;

    auto fit_circle(const aimer::math::CircleF& circle) const -> aimer::math::CircleF;

private:
    auto get_param_k() const -> double;

    aimer::CoordConverter* const converter;
    const double g { 9.8 }; // g = 9.8
    // const double shoot_angle;
    // const double v0;
    const aim::IdTLatencyAimCorrection aim;
    const double fire_t;
};

/**
 * @brief 自动模拟子弹和计算实际发射和理想发射误差
 *
 */
struct IdPos {
    int id;
    Eigen::Vector3d pos;
};

struct IdCircle {
    int id;
    aimer::math::CircleF circle;
};

// 存储被发射的子弹的模拟器
struct IdProj {
    int id;
    aim::ProjectileSimulator proj;
};

class AimCorrector {
public:
    AimCorrector(aimer::CoordConverter* const converter);
    auto add_aim(const aim::IdTLatencyAimCorrection& aim) -> void;
    // 电控检测到新的子弹被发射，以 last_shoot_id 的方式给出。
    // 若 id 切换时，我们尚无该 id 真正射出时的目标点。
    // 将 id 放入准备区
    auto update_bullet_id(const int& last_shoot_id) -> void;

    // 该函数并非只读，且根据 converter 自动获取时间。
    // 事实上，有些组件不得不获取四元数来继续计算。
    auto get_bullets() -> std::vector<aim::IdPos>;

    auto get_circles() -> std::vector<aim::IdCircle>;

    /** @brief 采样一次瞄准误差 */
    auto sample_aim_errors() -> void;

    auto get_aim_error() const -> Eigen::Vector2d;

private:
    // 对于一个子弹模拟器而言的 catch 写在哪里

    auto undistorted_circle(const aimer::math::CircleF& circle) -> aimer::math::CircleF;

    aimer::CoordConverter* const converter; // converter means NOW (current frame)
    int last_shoot_id = 0;
    aim::DetectBullet bullet_detector;
    aim::AimHistory aim_history;

    // Corrector 不得不包含模拟器实例，没办法
    // 实例 -- t --> 给出坐标，我转换成图像中的圆
    // simulator 捕获圆
    std::list<aim::IdProj> bullets;
    // 实例的管理

    // 未找到对应发射信息的子弹序号队列
    std::queue<int> pending_ids;
    // 实例的运用

    // 然而校正的响应有延迟。分别滤波 yaw 和 pitch
    aimer::SingleFilter<1> error_filters[2];
    // yaw, pitch
    std::deque<Eigen::Vector2d> error_angles;
};

} // namespace aimer::aim

#endif /* AIMER_AUTO_AIM_PREDICTOR_AIM_AIM_CORRECTOR_HPP */
