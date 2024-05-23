#include "aimer/auto_aim/predictor/motion/lmtd_top_model.hpp"
#include "aimer/auto_aim/predictor/motion/top_model.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"

#include "aimer/base/robot/coord_converter.hpp"
#include "base/debug/debug.hpp"

namespace aimer::lmtd {

struct Measure {
public:
    // z 用于接收. 由 x 推导到 z
    // z: 当前需求比较的装甲板，rotate: super_yaw 到需求装甲的 rotate
    // ekf 之与测量转观测量
    template<typename T>
    void operator()(const T x[N_X], T z[N_Z]) const {
        // x[6] 是 super 板的 yaw
        const T xyz_armor[3] = { x[0] + ceres::cos(x[6]) * x[8],
                                 x[2] + ceres::sin(x[6]) * x[8],
                                 x[4] };
        T ypd[3];
        aimer::math::ceres_xyz_to_ypd(xyz_armor, ypd);
        for (int i = 0; i < 3; i++) {
            z[i] = ypd[i];
        }
        // orien_yaw = orien_yaw
        z[3] = x[6];
    }

private:
};

// do the same as Measure
Eigen::Vector3d state_to_armor_pos(const State& state) {
    return Eigen::Vector3d { state[0] + std::cos(state[6]) * state[8],
                             state[2] + std::sin(state[6]) * state[8],
                             state[4] };
}

// 这个用来计算 ypd_v 的
Eigen::Vector3d state_to_armor_v(const State& state) {
    const Eigen::Vector3d center_v = { state[1], state[3], state[5] };
    const Eigen::Vector2d radius_norm = aimer::math::rotate({ 1.0, 0.0 }, state[6]);
    const Eigen::Vector2d theta_norm = aimer::math::rotate(radius_norm, M_PI / 2.0);
    const Eigen::Vector2d theta_v = state[7] * theta_norm * state[8];
    return { center_v[0] + theta_v[0], center_v[1] + theta_v[1], center_v[2] };
}

// fn state_to_center_pos => { 写不出来，需要结合 dz }
// fn state_to_center_v => { state[1, 3, 5] }
Eigen::Vector3d state_to_center_v(const State& state) {
    return { state[1], state[3], state[5] };
}

double state_to_zn_to_armor(const State& state, const aimer::CoordConverter* const converter) {
    return aimer::math::reduced_angle(state[6] - converter->get_camera_z_i_yaw() + M_PI);
}

struct Predict {
public:
    explicit Predict(const double& delta_t): delta_t(delta_t) {}
    template<typename T>
    void operator()(const T x_pre[N_X], T x_cur[N_X]) const {
        x_cur[0] = x_pre[0] + this->delta_t * x_pre[1];
        x_cur[1] = x_pre[1];
        x_cur[2] = x_pre[2] + this->delta_t * x_pre[3];
        x_cur[3] = x_pre[3];
        x_cur[4] = x_pre[4] + this->delta_t * x_pre[5];
        x_cur[5] = x_pre[5];
        x_cur[6] = x_pre[6] + this->delta_t * x_pre[7];
        x_cur[7] = x_pre[7];
        x_cur[8] = x_pre[8];
    }

private:
    double delta_t = 0.;
};

State state_predict(const State& state, const double& dt) {
    State res;
    res[0] = state[0] + dt * state[1];
    res[1] = state[1];
    res[2] = state[2] + dt * state[3];
    res[3] = state[3];
    res[4] = state[4] + dt * state[5];
    res[5] = state[5];
    res[6] = state[6] + dt * state[7];
    res[7] = state[7];
    res[8] = state[8];
    return res;
}

// [ArmorFilter]

State ArmorFilter::predict(const double& t) const {
    const double dt = t - this->t;
    const auto state = state_predict(this->state, dt);
    return state;
}

Eigen::Vector3d ArmorFilter::predict_pos(const double& t) const {
    const State state = this->predict(t);
    return lmtd::state_to_armor_pos(state);
}

Eigen::Vector3d ArmorFilter::predict_v(const double& t) const {
    const State state = this->predict(t);
    return lmtd::state_to_armor_v(state);
}

// [end of ArmorFilter]

State observation_to_init_state(const Observation& observation, const double& radius) {
    State init_x;
    Eigen::Vector3d armor_xyz = aimer::math::ypd_to_xyz(
        aimer::math::YpdCoord(observation[0], observation[1], observation[2])
    );
    const Eigen::Vector2d radius_norm = aimer::math::rotate({ 1.0, 0.0 }, observation[3]);
    // 这里还在 dz == 0 的初始化阶段所以是写得出来 center_xyz 的。其他地方由 State 不能直接推导 center
    Eigen::Vector3d center_xyz = aimer::top::prolonged_center(armor_xyz, radius, radius_norm, 0.0);
    init_x[0] = center_xyz[0];
    init_x[1] = 0.0;
    init_x[2] = center_xyz[1];
    init_x[3] = 0.0;
    init_x[4] = armor_xyz[2];
    init_x[5] = 0.0;
    init_x[6] = observation[3];
    init_x[7] = 0.0;
    init_x[8] = radius;
    return init_x;
}

std::vector<State> state_vec_to_direct_state_vec(
    const std::vector<State>& state_vec,
    const double& max_orientation_angle,
    aimer::CoordConverter* const converter
) {
    std::vector<State> direct_vec;
    for (const auto& state: state_vec) {
        if (std::abs(state_to_zn_to_armor(state, converter)) <= max_orientation_angle) {
            direct_vec.push_back(state);
        }
    }
    return direct_vec;
}

AimAndState choose_direct_aim(
    const std::vector<State>& direct_state_vec,
    const aimer::CoordConverter* const converter
) {
    State chosen_state = direct_state_vec[0];
    double min_swing_cost = DBL_MAX;
    for (const auto& direct: direct_state_vec) {
        const Eigen::Vector3d target_pos = state_to_armor_pos(direct);
        const auto aim_ypd = converter->target_pos_to_aim_ypd(target_pos);
        const double swing_cost = converter->aim_swing_cost(aim_ypd);
        if (swing_cost < min_swing_cost) {
            chosen_state = direct;
            min_swing_cost = swing_cost;
        }
    }
    // 把他转换成 aim..
    const Eigen::Vector3d target_pos = state_to_armor_pos(chosen_state);
    const auto shoot_param = converter->target_pos_to_shoot_param(target_pos);
    const auto aim_ypd = converter->aim_xyz_i_to_aim_ypd(shoot_param.aim_xyz_i_barrel);
    const auto armor_v = state_to_armor_v(chosen_state);
    const auto aim_ypd_v = converter->get_camera_ypd_v(target_pos, armor_v);
    const auto aim = aimer::AimInfo { aim_ypd, aim_ypd_v, shoot_param, ::ShootMode::TRACKING };
    return AimAndState { aim, chosen_state };
}

// 只可以在角速度较大的时候调用哦
AimAndState choose_indirect_aim(
    const std::vector<State>& indirect_state_vec,
    const double& max_orientation_angle,
    const double& max_out_error,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) {
    const double w = indirect_state_vec[0][7];
    // "wait" is the angle where you should wait for the next armor to come (emerge)
    const double zn_to_wait = w > 0.0 ? -max_orientation_angle : +max_orientation_angle;
    double min_armor_to_wait = DBL_MAX;
    State indirect_aim_state = indirect_state_vec[0];
    for (const auto& state: indirect_state_vec) {
        const double max_out_angle =
            enemy_state->get_sample_armor_ref().width / 2.0 * max_out_error / state[8];
        const double zn_to_armor = state_to_zn_to_armor(state, converter);
        // 值域 (-max_out_angle, M_PI - max_out_angle)
        const double armor_to_wait =
            aimer::math::reduced_angle(
                (w > 0.0 ? zn_to_wait - zn_to_armor : zn_to_armor - zn_to_wait) - M_PI
                + max_out_angle
            )
            + M_PI - max_out_angle;
        if (armor_to_wait < min_armor_to_wait) {
            min_armor_to_wait = armor_to_wait;
            indirect_aim_state = state;
        }
    }
    // 把他转换成 aim
    // 这里 t 传入 0.0
    const auto armor_filter = ArmorFilter { indirect_aim_state, 0.0 };
    const double time_to_emerge = min_armor_to_wait / std::abs(w);
    const auto emerge_state = armor_filter.predict(time_to_emerge);
    const auto aim = [&]() {
        const Eigen::Vector3d emerge_pos = state_to_armor_pos(emerge_state);
        const aimer::ShootParam shoot_param = converter->target_pos_to_shoot_param(emerge_pos);
        const aimer::math::YpdCoord ypd =
            converter->aim_xyz_i_to_aim_ypd(shoot_param.aim_xyz_i_barrel);
        const Eigen::Vector3d center_v = state_to_center_v(emerge_state);
        // 这是指过去等待，不进行 theta_v 的跟随
        aimer::math::YpdCoord ypd_v = converter->get_camera_ypd_v(emerge_pos, center_v);
        // 这个 aim 供后面计算 ShootMode 参考，ShootMode 置空
        return aimer::AimInfo { ypd, ypd_v, shoot_param, ::ShootMode::TRACKING };
    }();
    // aim 瞄准是 emerge pos
    // 装甲板实际状态是 indirect_aim_state
    return AimAndState { aim, indirect_aim_state };
}

AimAndState state_vec_to_aim(
    const std::vector<State>& state_vec,
    const double& max_orientation_angle,
    const double& max_out_error,
    const bool& allow_indirect,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) {
    const std::vector<State> direct_vec =
        state_vec_to_direct_state_vec(state_vec, max_orientation_angle, converter);
    if (direct_vec.empty()) {
        if (!allow_indirect) {
            return { aimer::AimInfo::idle(), State {} };
        }
        return choose_indirect_aim(
            state_vec,
            max_orientation_angle,
            max_out_error,
            converter,
            enemy_state
        );
    }
    return choose_direct_aim(direct_vec, converter);
};

// #before

// 那么难题是如何选择装甲板呢？
// 像 SimpleTopModel 那样再维护一个状态似乎不太可靠
// - 选择角度较小的：角度太大的 pnp dis 不准
// - 选择左边的：防止两个装甲板之间反复横跳
void TopModel::init(aimer::CoordConverter* const converter, aimer::EnemyState* const enemy_state) {
    this->predict_t = converter->get_img_t();
    this->update_t = converter->get_img_t();
    const auto observation_and_r_and_id =
        this->get_observation_and_r_and_id(converter, enemy_state);
    const auto init_x = observation_to_init_state(
        std::get<0>(observation_and_r_and_id),
        enemy_state->get_default_radius()
    );
    this->ekf.init_x(init_x);
    this->dz = 0.0;
    this->another_radius = enemy_state->get_default_radius();
    this->tracked_armor_id = std::get<2>(observation_and_r_and_id);
    this->top_level = 0;
}

void TopModel::update(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) {
    const std::vector<aimer::ArmorData>& armors = enemy_state->get_armor_data_ref();
    if (not(1u <= armors.size() && armors.size() <= size_t(enemy_state->get_max_seen())
            && armors.size() <= 2u))
    {
        return;
    }
    // armors is no longer used after this line
    if (not this->credit(converter)) {
        // 不可信，意味着预测是毫无意义的
        // 第一次进入这个 update 也会 init
        this->init(converter, enemy_state);
        return;
    }
    {
        const double dt = converter->get_img_t() - this->predict_t;
        const auto get_q = [](const std::string& name) {
            return base::get_param<double>("auto-aim.lmtd-top-model.q." + name);
        };
        const double q_pos_x = get_q("pos.x.in-1s") * dt;
        const double q_pos_v = get_q("pos.v.in-1s") * dt;
        const double q_orientation_x = get_q("orientation-yaw.x.in-1s") * dt;
        const double q_orientation_v = get_q("orientation-yaw.v.in-1s") * dt;
        const double q_r = get_q("r");
        const std::vector<double> q_vec = { q_pos_x,         q_pos_v,         q_pos_x,
                                            q_pos_v,         q_pos_x,         q_pos_v,
                                            q_orientation_x, q_orientation_v, q_r };
        this->ekf.predict_forward(Predict(dt), aimer::math::vec_x_to_mat_xx<N_X>(q_vec));
        this->predict_t = converter->get_img_t();
    }
    // 首先计算被追踪的 armor 对应的 观测量 Z
    const auto tracked_observation_and_r_and_id =
        this->get_observation_and_r_and_id(converter, enemy_state);
    const auto tracked_observation = std::get<0>(tracked_observation_and_r_and_id);
    const auto tracked_r = std::get<1>(tracked_observation_and_r_and_id);
    const int tracked_id = std::get<2>(tracked_observation_and_r_and_id);

    // # 由于跳转的可能性
    // 而且这里跳转时，之前跟踪的 id 已经彻底丢失，所以采用展开的方法
    // [根据 filter.yaw 展开装甲板可能的 N 个位置]
    // 当 id 没有丢失时肯定没有跳转
    if (tracked_id != this->tracked_armor_id) {
        // ... jumping...
        const Eigen::Vector3d tracked_xyz = aimer::math::ypd_to_xyz(aimer::math::YpdCoord(
            tracked_observation[0],
            tracked_observation[1],
            tracked_observation[2]
        ));
        const State predicted_state = this->ekf.get_x();
        const double state_orientation_yaw = predicted_state[6];
        const int armors_num = enemy_state->get_armors_num();

        int most_like_index = 0;
        double most_like_orientation_yaw_diff = DBL_MAX;
        double possible_orientation_yaw[armors_num];
        for (int i = 0; i < armors_num; i++) {
            possible_orientation_yaw[i] =
                aimer::math::reduced_angle(state_orientation_yaw + i * (2.0 * M_PI / armors_num));
            const double yaw_diff = std::abs(
                aimer::math::reduced_angle(possible_orientation_yaw[i] - tracked_observation[3])
            );
            if (yaw_diff < most_like_orientation_yaw_diff) {
                most_like_index = i;
                most_like_orientation_yaw_diff = yaw_diff;
            }
        }

        if (most_like_index != 0) {
            auto fixed_state = predicted_state;
            // 四个装甲板车有两种半径
            if (armors_num == 4 && most_like_index % 2 == 1) {
                // 出现了跳转捏
                std::swap(fixed_state[8], this->another_radius);
                // after(fixed_z + dz) == before(tracked_z + fixed_z - tracked_z)
                // == before(fixed_z)
                this->dz = fixed_state[4] - tracked_xyz[2];
                // [修复 dz]
                const double dz_abs_max =
                    ::base::get_param<double>("auto-aim.lmtd-top-model.fix.dz.abs-max");
                this->dz = std::clamp(this->dz, -dz_abs_max, +dz_abs_max);
                fixed_state[4] = tracked_xyz[2];
                base::println("reset z {:.3f}", fixed_state[4]);
            }
            // 滤波器内速度太慢时无法得出任何靠谱的装甲板角度，以下写法爆炸了
            // fixed_state[6] = possible_orientation_yaw[most_like_index];
            fixed_state[6] = tracked_observation[3];
            this->ekf.set_x(fixed_state);
        }
    }

    const auto inner_observation = this->ekf.measure(Measure()).y_e;
    auto fixed_observation = tracked_observation;
    // yaw 在 -pi 到 pi 处会突变，通过下面这个方法试图解决
    // 注意到正常旋转时，切换装甲板时 yaw 的处理时强制设置的，所以不用担心
    // inner yaw 无限增长的问题
    fixed_observation[3] =
        aimer::math::get_closest_angle(fixed_observation[3], inner_observation[3]);
    fixed_observation[0] =
        aimer::math::get_closest_angle(fixed_observation[0], inner_observation[0]);

    {
        const auto xyz = aimer::math::ypd_to_xyz(
            { fixed_observation[0], fixed_observation[1], fixed_observation[2] }
        );
        base::println(
            "{:.3f} {:.3f} {:.3f}",
            xyz[0],
            xyz[2],
            aimer::math::rad_to_deg(fixed_observation[3])
        );
    }

    // 修正内部状态后
    // [观测更新]
    base::println("#before update z zv {:.4f} {:.4f}", this->ekf.get_x()[4], this->ekf.get_x()[5]);
    this->ekf
        .update_forward(Measure(), fixed_observation, aimer::math::array_to_diag_mat(tracked_r));
    base::println("#after  update z zv {:.4f} {:.4f}", this->ekf.get_x()[4], this->ekf.get_x()[5]);
    this->update_t = converter->get_img_t();
    this->tracked_armor_id = tracked_id;

    // [更新后修复]
    // [更新后修复.前哨站半径已知]
    if (enemy_state->get_enemy_type() == aimer::EnemyType::OUTPOST) {
        auto fixed_state = this->ekf.get_x();
        fixed_state[8] = enemy_state->get_default_radius();
        this->ekf.set_x(fixed_state);
    }
    // [更新后修复.防止半径求解误差过大]
    {
        using ::base::get_param;
        auto fixed_state = this->ekf.get_x();
        fixed_state[8] = std::clamp(
            fixed_state[8],
            get_param<double>("auto-aim.lmtd-top-model.fix.radius.min"),
            get_param<double>("auto-aim.lmtd-top-model.fix.radius.max")
        );
        // pnp 观测的 z 存在巨大误差，紧凑跟随时可能会产生明显的 z 速度
        if (base::get_param<bool>("auto-aim.lmtd-top-model.fix.no-z-axis-motion")) {
            fixed_state[5] = 0.0;
        }
        base::println("z v: {}", fixed_state[5]);
        this->ekf.set_x(fixed_state);
    }

    // [更新后修复.前哨站不移动的话可以优化]
    {
        if (enemy_state->get_enemy_type() == aimer::EnemyType::OUTPOST
            && base::get_param<bool>(
                "auto-aim.lmtd-top-model.fix.you-are-sure-that-the-outpost-will-not-move-in-a-continuous-observation"
            ))
        {
            auto fixed_state = this->ekf.get_x();
            fixed_state[1] = 0.0;
            fixed_state[3] = 0.0;
            fixed_state[5] = 0.0;
            this->ekf.set_x(fixed_state);
        }
    }

    // [反陀螺模式判断更新]
    this->update_top_level();

    {
        // debug
        using ::base::print;
        using ::base::println;
        using ::base::PrintMode;
        const auto state = this->ekf.get_x();
        std::string state_str = "";
        for (int i = 0; i < N_X; i++) {
            if (i == 6 || i == 7) {
                state_str += fmt::format("{:7.2f}", aimer::math::rad_to_deg(state[i]));

            } else {
                state_str += fmt::format("{:7.3f}", state[i]);
            }
        }

        // print_info("{} {}", this->tracked_armor_id, state_str);
    }
}

bool TopModel::credit(aimer::CoordConverter* const converter) const {
    const double credit_dt = ::base::get_param<double>("auto-aim.lmtd-top-model.credit-dt");
    return converter->get_img_t() - this->update_t <= credit_dt;
}

std::vector<PredictedArmor> TopModel::predict_armors(
    const double& t,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) const {
    const auto predicted_state = this->ekf.predict(Predict(t - this->predict_t)).x_p;
    const Eigen::Vector2d center_xy = { predicted_state[0], predicted_state[2] };
    double orientation_yaw = predicted_state[6];
    double radius = predicted_state[8];
    double another_r = this->another_radius;
    double armor_z = predicted_state[4];
    double dz = this->dz;
    std::vector<PredictedArmor> armors;
    for (int i = 0; i < enemy_state->get_armors_num(); i++) {
        const Eigen::Vector3d armor_pos = { center_xy[0] + std::cos(orientation_yaw) * radius,
                                            center_xy[1] + std::sin(orientation_yaw) * radius,
                                            armor_z };
        const double orientation_pitch = enemy_state->get_armor_pitch();
        const aimer::ArmorType type = enemy_state->get_sample_armor_ref().type;
        armors.emplace_back(PredictedArmor { armor_pos, orientation_yaw, orientation_pitch, type });
        // next
        orientation_yaw += 2.0 * M_PI / enemy_state->get_armors_num();
        if (enemy_state->get_armors_num() == 4) {
            std::swap(radius, another_r);
            armor_z += dz;
            dz = -dz;
        }
    }
    return armors;
}

void TopModel::draw_armors(
    cv::Mat& img,
    const double& t,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) const {
    const auto predicted_armor = this->predict_armors(t, converter, enemy_state);
    const cv::Scalar color = this->top_level == 0 ? cv::Scalar { 127, 127, 127 }
        : this->top_level == 1                    ? cv::Scalar { 151, 128, 255 }
                                                  : cv::Scalar { 235, 206, 135 };
    for (const auto& armor: predicted_armor) {
        const double z_to_armor =
            aimer::math::reduced_angle(armor.orientation_yaw - converter->get_camera_z_i_yaw());
        const std::vector<cv::Point2f> four_corners_points = aimer::top::radial_armor_pts(
            armor.pos,
            armor.type,
            armor.orientation_pitch,
            z_to_armor,
            converter
        );
        aimer::debug::draw_lines(img, four_corners_points, color, 2, true);
    }
}

int TopModel::get_top_level() const {
    return this->top_level;
}

std::vector<ArmorFilter> TopModel::get_armor_filters(aimer::EnemyState* const enemy_state) const {
    const int armors_num = enemy_state->get_armors_num();
    // 直接获取目前的状态做滤波器就行
    const auto one_armor_state = this->ekf.predict(Predict(0.0)).x_p;
    double orientation_yaw = one_armor_state[6];
    double radius = one_armor_state[8];
    double another_radius = this->another_radius;
    double armor_z = one_armor_state[4];
    double dz = this->dz;
    std::vector<ArmorFilter> filters;
    for (int i = 0; i < armors_num; i++) {
        State armor_state;
        armor_state[0] = one_armor_state[0];
        armor_state[1] = one_armor_state[1];
        armor_state[2] = one_armor_state[2];
        armor_state[3] = one_armor_state[3];
        armor_state[4] = armor_z;
        armor_state[5] = one_armor_state[5];
        armor_state[6] = orientation_yaw;
        armor_state[7] = one_armor_state[7];
        armor_state[8] = radius;
        filters.emplace_back(armor_state, this->predict_t);
        // 切换到下一块装甲板
        orientation_yaw += 2.0 * M_PI / armors_num;
        if (armors_num == 4) {
            std::swap(radius, another_radius);
            armor_z += dz;
            dz = -dz;
        }
    }
    return filters;
}

aimer::AimInfo TopModel::get_aim(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) const {
    using ::base::get_param;
    if (!this->credit(converter)) {
        return aimer::AimInfo::idle();
    }
    // todo
    const int armors_num = enemy_state->get_armors_num();
    std::vector<ArmorFilter> filters = this->get_armor_filters(enemy_state);
    // ... num
    std::vector<State> water_gun_hit_state_vec = {};
    for (const auto& filter: filters) {
        const double water_gun_hit_time = converter->filter_to_prediction_time(filter);
        const State that_state = filter.predict(water_gun_hit_time);
        water_gun_hit_state_vec.emplace_back(that_state);
    }
    // [选择应该命令电机转向哪个装甲板]
    // 如果有 angle 以内的，直接选择其中需要转动角最小的那个就行
    // 否则需要计算 emerging pos（即将出现在合适打击的位置）
    const double max_orientation_angle = aimer::math::deg_to_rad([&]() {
        if (this->top_level == 0) {
            if (armors_num == 4) {
                return get_param<double>(
                    "auto-aim.lmtd-top-model.aim.top0.max-orientation-angle.armors-4"
                );
            }
            return get_param<double>(
                "auto-aim.lmtd-top-model.aim.top0.max-orientation-angle.armors-other"
            );
        }
        return get_param<double>(
            fmt::format("auto-aim.lmtd-top-model.aim.top{}.max-orientation-angle", this->top_level)
        );
    }());
    const double max_out_error = get_param<double>(
        fmt::format("auto-aim.lmtd-top-model.aim.top{}.max-out-error", this->top_level)
    );
    const bool allow_indirect = this->top_level > 0;
    const auto water_gun_hit_aim_and_state = state_vec_to_aim(
        water_gun_hit_state_vec,
        max_orientation_angle,
        max_out_error,
        allow_indirect,
        converter,
        enemy_state
    );
    if (water_gun_hit_aim_and_state.aim.shoot == ::ShootMode::IDLE) {
        return aimer::AimInfo::idle();
    }
    std::vector<State> command_hit_state_vec = {};
    for (const auto& filter: filters) {
        const double command_hit_time = converter->filter_to_hit_time(filter);
        const State that_state = filter.predict(command_hit_time);
        command_hit_state_vec.emplace_back(that_state);
    }
    const auto command_hit_aim_and_state = state_vec_to_aim(
        command_hit_state_vec,
        max_orientation_angle,
        max_out_error,
        /*allow_indirect=*/true,
        converter,
        enemy_state
    );
    const bool you_had_better_shoot_at_this_command = [&]() {
        // [如果跟随误差太大就不发]
        const double max_tracking_error =
            ::base::get_param<double>("auto-aim.lmtd-top-model.aim.max-tracking-error");
        if (converter->aim_error_exceeded(
                water_gun_hit_aim_and_state.aim.ypd,
                enemy_state->get_sample_armor_ref(),
                max_tracking_error,
                state_to_zn_to_armor(water_gun_hit_aim_and_state.state, converter),
                enemy_state->get_armor_pitch()
            ))
        {
            return false; // don't send shoot command
        }
        // [如果 command_hit 的时候在旋转枪口那就不能发]
        // 首先判断 water_gun_hit 到 command_hit 是否在回转
        // 如果是回转的，可以算出开始回转的时间点
        // 也可以算出回转的角度有多大（有一个近似，就是假设转到 command_hit 的角度）
        // 然后根据 angle_to_rotate_time 算出回转结束的时间点
        // 如果 command_hit time 在两个之间就不能发
        // [.注意] 区分枪口的 yaw 和装甲板的 yaw。两个在不同地方使用

        // 在 top_level == 0 时角速度不可信，这里不能用来算关键时间点
        if (this->top_level > 0) {
            const auto angle_to_rotate_time = [](const double& angle) -> double {
                const double a =
                    base::get_param<double>("auto-aim.lmtd-top-model.aim.angle-to-rotate-time.a");
                const double b =
                    base::get_param<double>("auto-aim.lmtd-top-model.aim.angle-to-rotate-time.b");
                // 注意这里线性函数的参数是角度制哦
                return a * aimer::math::rad_to_deg(angle) + b;
            };
            const double w_water_gun_hit = water_gun_hit_aim_and_state.state[7];
            const double armor_rotate_water_gun_hit_to_command_hit = aimer::math::reduced_angle(
                command_hit_aim_and_state.state[6] - water_gun_hit_aim_and_state.state[6]
            );

            // 这段逻辑很复杂但是看起来运行的还挺正常
            // base::print_info(
            //     "---\nw: {}\nr: {}",
            //     aimer::math::rad_to_deg(w_water_gun_hit),
            //     aimer::math::rad_to_deg(armor_rotate_water_gun_hit_to_command_hit)
            // );

            if (std::signbit(w_water_gun_hit)
                != std::signbit(armor_rotate_water_gun_hit_to_command_hit))
            {
                // 小心，这里没有预防 water_gun_hit 装甲板超过 max_orientation_angle 的情况
                const double zn_to_armor_water_gun_hit =
                    state_to_zn_to_armor(water_gun_hit_aim_and_state.state, converter);
                // const double zn_to_armor_command_hit =
                //     state_to_zn_to_armor(command_hit_aim_and_state.state, converter);
                const double zn_to_where_you_should_rotate_back =
                    w_water_gun_hit > 0.0 ? +max_orientation_angle : -max_orientation_angle;
                const double armor_water_gun_hit_to_rotate_back = aimer::math::reduced_angle(
                    zn_to_where_you_should_rotate_back - zn_to_armor_water_gun_hit
                );
                // const double angle_rotate_back = aimer::math::reduced_angle(
                //     zn_to_armor_command_hit - zn_to_where_you_should_rotate_back
                // );

                // 这里的 time 是正常原点的时间
                const double time_water_gun_hit = converter->get_prediction_time(
                    state_to_armor_pos(water_gun_hit_aim_and_state.state)
                );
                const double time_command_hit =
                    converter->get_hit_time(state_to_armor_pos(command_hit_aim_and_state.state));
                const double time_start_rotating_back =
                    time_water_gun_hit + armor_water_gun_hit_to_rotate_back / w_water_gun_hit;
                const auto filter =
                    ArmorFilter { water_gun_hit_aim_and_state.state, time_water_gun_hit };
                const auto pos_when_start_rotating_back =
                    filter.predict_pos(time_start_rotating_back);
                const auto aim_when_start_rotating_back =
                    converter->target_pos_to_aim_ypd(pos_when_start_rotating_back);
                const double yaw_barrel_rotate_back = aimer::math::reduced_angle(
                    command_hit_aim_and_state.aim.ypd.yaw - aim_when_start_rotating_back.yaw
                );
                const double time_end_rotating_back = time_start_rotating_back
                    + angle_to_rotate_time(std::abs(yaw_barrel_rotate_back));
                // base::print_info(
                //     "***\nrotate: {} {}",
                //     yaw_barrel_rotate_back,
                //     angle_to_rotate_time(std::abs(yaw_barrel_rotate_back))
                // );
                // 完蛋了我不知道 command_hit_time
                // 无所谓，原地算好了

                // base::print_info(
                //     "***\ns: {}\ne: {}\nh: {}",
                //     time_start_rotating_back,
                //     time_end_rotating_back,
                //     time_command_hit
                // );
                if (time_start_rotating_back < time_command_hit
                    && time_command_hit < time_end_rotating_back)
                {
                    return false; // don't send shoot command
                }
            }
        }
        if (converter->aim_error_exceeded(
                command_hit_aim_and_state.aim.ypd,
                converter->target_pos_to_aim_ypd(state_to_armor_pos(command_hit_aim_and_state.state)
                ),
                enemy_state->get_sample_armor_ref(),
                max_out_error,
                state_to_zn_to_armor(command_hit_aim_and_state.state, converter),
                enemy_state->get_armor_pitch()
            ))
        {
            return false; // don't send shoot command
        }
        return true; // send shoot command
    }();
    auto aim_with_shoot_command = water_gun_hit_aim_and_state.aim;
    aim_with_shoot_command.shoot =
        you_had_better_shoot_at_this_command ? ::ShootMode::SHOOT_NOW : ::ShootMode::TRACKING;
    return aim_with_shoot_command;
}

std::tuple<Observation, std::array<double, N_Z>, int> TopModel::get_observation_and_r_and_id(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) const {
    // 还是仿照 SimpleTopModel 吧
    const std::vector<aimer::ArmorData>& armors = enemy_state->get_armor_data_ref();
    const int tracked_index = [&]() {
        if (armors.size() == 1) {
            return 0;
        }
        float max_area = DBL_MIN;
        int max_area_index = -1;
        float tracked_id_area = 0.0;
        int tracked_id_index = -1;
        // 下面要找到最大面积的和状态机中追踪 id 的面积哦
        // 但后者不一定能找到
        for (size_t i = 0; i < armors.size(); i++) {
            const double area = armors[i].info.area();
            if (area > max_area) {
                max_area_index = i;
                max_area = area;
            }
            if (armors[i].id == this->tracked_armor_id) {
                tracked_id_index = i;
                tracked_id_area = area;
            }
        }
        const double keep_tracking_ratio =
            ::base::get_param<double>("auto-aim.lmtd-top-model.keep-tracking-area-ratio");
        if (tracked_id_index != -1 && tracked_id_area >= keep_tracking_ratio * max_area) {
            return tracked_id_index;
        } else {
            return max_area_index;
        }
    }();
    const double z_to_armor_exp = [&]() {
        if (this->tracked_armor_id == armors[tracked_index].id) {
            const double orientation_yaw =
                this->ekf.predict(Predict(converter->get_img_t() - this->predict_t)).x_p[6];
            return aimer::math::reduced_angle(orientation_yaw - converter->get_camera_z_i_yaw());
        } else {
            return M_PI / 4.0 * 3.0; // 135 度是装甲板在左半边 45 度
        }
    }();
    const auto get_r = [](const std::string& name) {
        return ::base::get_param<double>("auto-aim.lmtd-top-model.r." + name);
    };
    const auto orientation_yaw_and_its_r = [&]() -> std::tuple<double, double> {
        const double angle_between_armors = 2.0 * M_PI / enemy_state->get_armors_num();
        // 朝向角与相机 z 轴反方向的夹角在该角度之内的装甲板几乎必定被观察
        const double must_see_angle = M_PI / 4.0;
        const double must_not_see_angle = M_PI / 2.0;
        const auto z_to_armor_and_r = [&]() -> std::tuple<double, double> {
            if (armors.size() == 1) {
                const double z_to_armor_min = std::max(
                    M_PI - must_not_see_angle,
                    M_PI + must_see_angle - angle_between_armors
                );
                const double z_to_armor_max = std::min(
                    M_PI + must_not_see_angle,
                    M_PI - must_see_angle + angle_between_armors
                );
                // 当 3 装甲板时，这两个角是：朝左 / 朝右 75 度
                return { aimer::top::fit_single_z_to_v(
                             armors[tracked_index],
                             z_to_armor_exp,
                             z_to_armor_min,
                             z_to_armor_max,
                             converter
                         ),
                         get_r("orientation-yaw.single") };
            } else {
                // armors.size() == 2
                const double z_to_left = aimer::top::fit_double_z_to_l(
                    armors,
                    z_to_armor_exp,
                    M_PI - must_not_see_angle,
                    M_PI + must_not_see_angle - angle_between_armors,
                    converter
                );
                const double r = get_r("orientation-yaw.double");
                if (tracked_index == 0) {
                    return { z_to_left, r };
                } else {
                    // tracked_index == 1 根据 EnemyState 中的排序，[1] 在右边
                    return { z_to_left + angle_between_armors, r };
                }
            }
        }();
        return { aimer::math::reduced_angle(
                     converter->get_camera_z_i_yaw() + std::get<0>(z_to_armor_and_r)
                 ),
                 std::get<1>(z_to_armor_and_r) };
    }();
    const aimer::math::YpdCoord ypd = aimer::math::xyz_to_ypd(armors[tracked_index].info.pos);
    const Observation observation = { ypd.yaw,
                                      ypd.pitch,
                                      ypd.dis,
                                      std::get<0>(orientation_yaw_and_its_r) };
    const std::array<double, N_Z> rs = { get_r("yaw"),
                                         get_r("pitch"),
                                         get_r("dis.at-1m") * std::pow(ypd.dis, 4.0),
                                         std::get<1>(orientation_yaw_and_its_r) };
    return { observation, rs, armors[tracked_index].id };
}

void TopModel::update_top_level() {
    const auto credible_state = this->ekf.get_x();
    const double credible_abs_w = std::abs(credible_state[7]);
    // 最大 top 等级是 2 写死了
    // 是这样的
    // 0 级：基本上直接跟随就行
    // 1 级：引入 indirect
    // 2 级：转得太快跟不上，改成往中间打
    const double next_level_activate_w = this->top_level == 2
        ? DBL_MAX
        : aimer::math::deg_to_rad(::base::get_param<double>(
              fmt::format("auto-aim.lmtd-top-model.aim.top{}.activate-w", this->top_level + 1)
          ));
    const double deactivate_w = this->top_level == 0
        ? DBL_MIN
        : aimer::math::deg_to_rad(::base::get_param<double>(
              fmt::format("auto-aim.lmtd-top-model.aim.top{}.deactivate-w", this->top_level)
          ));

    if (credible_abs_w >= next_level_activate_w) {
        this->top_level += 1;
    } else if (credible_abs_w < deactivate_w) {
        this->top_level -= 1;
    }
}
} // namespace aimer::lmtd
