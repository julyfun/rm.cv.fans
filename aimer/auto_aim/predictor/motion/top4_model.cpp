#include "aimer/auto_aim/predictor/motion/top4_model.hpp"

#include <algorithm>

#include "aimer/base/debug/debug.hpp"

namespace aimer::top::top4 {

/** @class AbArmor */

auto AbManager::get_ab() const -> top4::AbArmor {
    return top4::AbArmor(this->length_filter.predict(0.)(0, 0), this->z_filter.predict(0.)(0, 0));
}

void AbManager::update(const Eigen::Vector3d& center, const top4::AbArmor& d) {
    auto get_param = [](const std::string& name) -> double {
        return ::base::get_param<double>("auto-aim.top-model.ab." + name);
    };
    double dis = center.norm();
    this->length_filter.update(
        std::clamp(d.length, get_param("length.min"), get_param("length.max")),
        0.,
        { 1. },
        { get_param("length.r-at-1m") * dis * dis });
    this->z_filter.update(
        std::clamp(d.z_plus, -get_param("z.abs-max"), get_param("z.abs-max")),
        0.,
        { 1. },
        { get_param("z.r-at-1m") * dis * dis });
}

double AbManager::get_cost(const top4::AbArmor& d) const {
    top4::AbArmor res = this->get_ab();
    return std::fabs(res.length - d.length) * 3. + std::fabs(res.z_plus - d.z_plus) * 10.;
}

/** @class IdDirectionJudger */

top::TopDirection IdDirectionJudger::get() const {
    return this->direction;
}

void IdDirectionJudger::update(const std::vector<aimer::ArmorData>& armors) {
    if (this->last_armors.size() == 1 && armors.size() == 2) {
        if (this->last_armors[0].id == armors[0].id) {
            this->direction = top::TopDirection::CW;
        } else if (this->last_armors[0].id == armors[1].id) {
            this->direction = top::TopDirection::CCW;
        }
        // 其他情况非法，保持原判断
    } else if (this->last_armors.size() == 2 && armors.size() == 1) {
        if (this->last_armors[0].id == armors[0].id) {
            this->direction = top::TopDirection::CCW;
        } else if (this->last_armors[1].id == armors[0].id) {
            this->direction = top::TopDirection::CW;
        }
    }
    this->last_armors = armors;
}

/** @class RotateCounter */

RotateCounter::RotateCounter(const double& init_radius) {
    this->ab_manager[0].update(Eigen::Vector3d(1., 0., 0.), top4::AbArmor(init_radius, 0.));
    this->ab_manager[1].update(Eigen::Vector3d(1., 0., 0.), top4::AbArmor(init_radius, 0.));
}

top4::TopPattern RotateCounter::get_pattern() const {
    return this->pattern;
}

int RotateCounter::get_active_rotate() const {
    return this->active_rotate;
}

top4::AbArmor RotateCounter::get_ab(const int& index) const {
    return this->ab_manager[index].get_ab();
}

double RotateCounter::get_w() const {
    return this->theta_s_filter.get_w();
}

// double get_theta_l() const {
//   double theta_s = this->theta_s_filter.get_angle();
//   return aimer::math::reduced_angle(theta_s - this->rotate * M_PI / 2.);
// }

// 外部传进的数据宽容对待，我给外部的数据严格优秀
double RotateCounter::predict_theta_l(const double& t) const {
    double theta_s = this->theta_s_filter.predict_angle(t);
    return aimer::math::reduced_angle(theta_s - this->rotate * M_PI / 2.);
}

bool RotateCounter::w_active() const { // 旋转状态
    return std::fabs(this->get_w())
        > base::get_param<double>("auto-aim.top-model.active-w") / 180. * M_PI;
}

bool RotateCounter::w_inactive() const {
    return std::fabs(this->get_w())
        <= base::get_param<double>("auto-aim.top-model.inactive-w") / 180. * M_PI;
}

bool RotateCounter::top_active() const {
    return !this->w_inactive()
        && std::fabs(this->get_active_rotate())
        >= base::get_param<int64_t>("auto-aim.top-model.top4.active-rotate");
}

void RotateCounter::update_theta_l(const double& theta_l, const double& t, const double& R) {
    // 不可用本来就错的一塌糊涂的 theta_s 反向修正 rotate
    this->theta_s_filter.update(theta_l + this->rotate * M_PI / 2., t, R);
}

void RotateCounter::update_rotate(const std::vector<aimer::ArmorData>& armors) {
    this->id_direction_judger.update(armors); // 可认为是 virtual direction
    top::TopDirection id_direction = this->id_direction_judger.get();
    bool top_pre_active = this->top_active();
    if (armors.size() == 1 && this->pattern != top4::TopPattern::A
        && this->pattern != top4::TopPattern::B) {
        if (this->pattern == top4::TopPattern::AB) {
            if (id_direction == top::TopDirection::CCW) {
                this->pattern = top4::TopPattern::A;
            } else {
                this->pattern = top4::TopPattern::B;
                --this->rotate;
                --this->active_rotate;
                // active_rotate +1 -1 左右旋会互相抵消
                // 判断时用 fabs(active_rotate)
            }
        } else {
            if (id_direction == top::TopDirection::CCW) {
                this->pattern = top4::TopPattern::B;
            } else {
                this->pattern = top4::TopPattern::A;
                --this->rotate;
                --this->active_rotate;
            }
        }
    } else if (
        armors.size() == 2 && this->pattern != top4::TopPattern::AB
        && this->pattern != top4::TopPattern::BA) {
        if (this->pattern == top4::TopPattern::A) {
            if (id_direction == top::TopDirection::CCW) {
                this->pattern = top4::TopPattern::BA;
                ++this->rotate;
                ++this->active_rotate;
            } else {
                this->pattern = top4::TopPattern::AB;
            }
        } else {
            if (id_direction == top::TopDirection::CCW) {
                this->pattern = top4::TopPattern::AB;
                ++this->rotate;
                ++this->active_rotate;
            } else {
                this->pattern = top4::TopPattern::BA;
            }
        }
    }
    if (top_pre_active) {
        if (this->w_inactive()) {
            this->active_rotate = 0; // 更易保持
        }
    } else {
        if (!this->w_active()) {
            this->active_rotate = 0; // 不易保持
        }
    }
}

// 根据长度和高低对 AB 进行检查和修正，仅在部分采样点这样做
void RotateCounter::update_ab(
    const Eigen::Vector3d& center,
    const std::vector<top4::AbArmor>& ab,
    const double& t) {
    double cost_ab = 0., cost_ba = 0.;
    for (int i = 0; i <= 1; ++i) {
        // 注意这边存储逻辑，传入的 ab vector 0 为图中左，1 为图中右
        // 而 ab_manager 中 [0] 为 A，[1] 为 B
        // 若认为图中 左 A 右 B，则代价为 0 ~ 0 + 1 ~ 1
        // 若认为图中 左 B 右 A，则代价为 0 ~ 1 + 1 ~ 0
        cost_ab += this->ab_manager[i].get_cost(ab[i]);
        cost_ba += this->ab_manager[i].get_cost(ab[1 - i]);
    }
    double cost = this->pattern == top4::TopPattern::AB ? cost_ab - cost_ba : cost_ba - cost_ab;
    this->pattern_fixer.update(cost, t);
    if (this->pattern_fixer.get_x_k1()(0, 0) > 0.) {
        this->pattern =
            this->pattern == top4::TopPattern::AB ? top4::TopPattern::BA : top4::TopPattern::AB;
        this->pattern_fixer.init();
        this->pattern_fixer.update(-cost, t);
    }
    if (this->pattern == top4::TopPattern::AB) {
        for (int i = 0; i <= 1; ++i) {
            this->ab_manager[i].update(center, ab[i]);
        }
    } else {
        for (int i = 0; i <= 1; ++i) {
            this->ab_manager[i].update(center, ab[1 - i]);
        }
    }
}

/** @class TopModel */

TopModel::TopModel(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state,
    const double& credit_time):
    converter(converter),
    state(state),
    credit_clock(converter, credit_time),
    counter(state->get_default_radius()) {}
// 少了少了，常数又少啦
// 输入时间 t，推断四块装甲板 t 时的位置
// 输入包括我机此刻信息（相机角度），并假定我机不变
// 保留

bool TopModel::active() const {
    return this->counter.top_active();
}

std::vector<TopArmor> TopModel::predict_armors(const double& t) const {
    Eigen::Vector3d center = this->center_filter.predict_pos(t);
    double theta = this->counter.predict_theta_l(t);
    int index = (this->counter.get_pattern() == top4::TopPattern::A
                 || this->counter.get_pattern() == top4::TopPattern::AB)
        ? 0
        : 1;
    Eigen::Vector2d norm2 { 1., 0. };
    norm2 = aimer::math::rotate(norm2, theta);
    double theta_zn = converter->get_camera_z_i_yaw() + M_PI;
    std::vector<TopArmor> res;
    for (int i = 0; i < 4; ++i) {
        double length = this->counter.get_ab(index).length;
        double z_plus = this->counter.get_ab(index).z_plus;
        Eigen::Vector3d pos { center(0, 0) + length * norm2(0, 0),
                              center(1, 0) + length * norm2(1, 0),
                              center(2, 0) + z_plus };
        res.emplace_back(pos, t, aimer::math::reduced_angle(theta - theta_zn), length, z_plus);
        norm2 = aimer::math::rotate(norm2, M_PI / 2.);
        index = index ^ 1;
        theta += M_PI / 2.;
    }
    return res;
}

// 类化
aimer::AimInfo TopModel::get_limit_aim() const {
    if (!this->credit_clock.credit()) {
        return aimer::AimInfo::idle();
    }
    // 目前，这里两个时间都是假定打中中心的时间，
    // 1. 这样逻辑简单一点
    // 2. 避免半径测不准的时候时机错乱
    // 这样逻辑上是有问题的，然而一定程度上被手测的延迟参数修正

    // 此时下达转动指令，control 时间点发射的子弹的击中时间
    std::vector<TopArmor> prediction_results =
        this->predict_armors(this->converter->filter_to_prediction_time(this->center_filter));
    // 此时下达开火指令，这一指令对应发射子弹的击中时间
    std::vector<TopArmor> hit_results =
        this->predict_armors(this->converter->filter_to_hit_time(this->center_filter));
    return top::get_top_limit_aim(
        this->center_filter,
        prediction_results,
        hit_results,
        this->counter.get_w(),
        this->converter,
        this->state);
}

void TopModel::update() { // 1 or 2
    const std::vector<aimer::ArmorData>& armors = this->state->get_armor_data_ref();
    if (!(armors.size() == 1 || armors.size() == 2)) {
        // 无法更新
        return;
    }
    this->credit_clock.update();

    this->counter.update_rotate(armors); // 遮挡判断机制优化后需要修改这个
    if (armors.size() == 2) {
        this->update_by_double_armors();
    } else { // size == 1
        this->update_by_single_armor();
    }

    aimer::debug::flask_aim << fmt::format("top_w {:.2f}", this->counter.get_w() / M_PI * 180.);
    if (this->counter.get_pattern() == top4::TopPattern::A) {
        aimer::debug::flask_aim << fmt::format(
            "len in {:.3f}|#{:.3f}",
            this->counter.get_ab(0).length,
            this->counter.get_ab(1).length);
    } else if (this->counter.get_pattern() == top4::TopPattern::AB) {
        aimer::debug::flask_aim << fmt::format(
            "len in {:.3f}|{:.3f}",
            this->counter.get_ab(0).length,
            this->counter.get_ab(1).length);
    } else if (this->counter.get_pattern() == top4::TopPattern::B) {
        aimer::debug::flask_aim << fmt::format(
            "len in {:.3f}|#{:.3f}",
            this->counter.get_ab(1).length,
            this->counter.get_ab(0).length);
    } else if (this->counter.get_pattern() == top4::TopPattern::BA) {
        aimer::debug::flask_aim << fmt::format(
            "len in {:.3f}|{:.3f}",
            this->counter.get_ab(1).length,
            this->counter.get_ab(0).length);
    }
}

// 本函数禁用 this 以保证函数性
auto TopModel::length_sampling(
    const std::vector<aimer::ArmorData>& armors,
    const double& z_to_l_exp,
    aimer::CoordConverter* const converter) const -> LengthSample {
    // 拿出纸笔画一下即可知道本函数思路
    // 双板距离都可信，进行最大化更新
    double z_to_l_fit = top::fit_double_z_to_l(armors, z_to_l_exp, M_PI / 2., M_PI, converter);

    // // 在核心采样点复现圆心并对长轴和短轴进行采样
    Eigen::Vector2d l_pos2 = Eigen::Vector2d(armors[0].info.pos(0, 0), armors[0].info.pos(1, 0)),
                    r_pos2 = Eigen::Vector2d(armors[1].info.pos(0, 0), armors[1].info.pos(1, 0));
    Eigen::Vector2d center2 = aimer::math::get_intersection(
        l_pos2,
        aimer::math::rotate(converter->get_camera_z_i2(), z_to_l_fit),
        r_pos2,
        aimer::math::rotate(converter->get_camera_z_i2(), z_to_l_fit + M_PI / 2.));
    Eigen::Vector3d center { center2(0, 0),
                             center2(1, 0),
                             (armors[0].info.pos(2, 0) + armors[1].info.pos(2, 0)) / 2. };
    double len_l = (l_pos2 - center2).norm();
    double len_r = (r_pos2 - center2).norm();

    // 绘制拟合 8 点
    {
        std::vector<std::vector<cv::Point2f>> double_pts =
            top::radial_double_pts(armors, z_to_l_fit, converter);
        // 绘制半径采样信息
        aimer::debug::flask_map << aimer::debug::poses_to_map_lines(
            { armors[0].info.pos, center, armors[1].info.pos },
            { 0, 255, 0 },
            false,
            aimer::debug::FLASK_MAP_THICKNESS);
    }

    return TopModel::LengthSample(
        center,
        z_to_l_fit,
        { top4::AbArmor(len_l, armors[0].info.pos(2, 0) - center(2, 0)),
          top4::AbArmor(len_r, armors[1].info.pos(2, 0) - center(2, 0)) });
}

// 返回 z_to_l_fit 角
double TopModel::double_angle_sampling(
    const std::vector<aimer::ArmorData>& armors,
    const double& z_to_l_exp,
    aimer::CoordConverter* const converter) const {
    double z_to_l_fit_by_l =
        top::fit_single_z_to_v(armors[0], z_to_l_exp, M_PI / 2., M_PI, converter);
    double z_to_l_fit_by_r = [&]() {
        double z_to_r_fit = top::fit_single_z_to_v(
            armors[1],
            z_to_l_exp + M_PI / 2,
            M_PI,
            M_PI / 2. * 3.,
            converter);
        return aimer::math::reduced_angle(z_to_r_fit - M_PI / 2.);
    }();
    // 若直接加权，需保证符号为正
    double z_to_l_fit_weight = aimer::math::get_weighted_angle(
        z_to_l_fit_by_l,
        armors[0].info.area(),
        z_to_l_fit_by_r,
        armors[1].info.area());
    return z_to_l_fit_weight;
}

void TopModel::update_by_double_armors() {
    const std::vector<aimer::ArmorData>& armors = this->state->get_armor_data_ref();
    if (aimer::math::get_ratio(armors[0].info.area(), armors[1].info.area())
        > base::get_param<double>("auto-aim.top-model.radius-sampling-area-ratio")) {
        TopModel::LengthSample length_sample = this->length_sampling(
            armors,
            aimer::math::reduced_angle(
                this->counter.predict_theta_l(this->converter->get_img_t())
                - this->converter->get_camera_z_i_yaw()),
            this->converter); // 包括角度取样

        aimer::debug::flask_aim << fmt::format(
            "zn_to_l[D]: {:.2f}",
            aimer::math::reduced_angle(length_sample.z_to_l_fit + M_PI) / M_PI * 180.);
        aimer::debug::flask_aim << fmt::format(
            "len sampLR {:.3f}|{:.3f}",
            length_sample.ab[0].length,
            length_sample.ab[1].length);

        this->counter.update_theta_l(
            length_sample.z_to_l_fit + this->converter->get_camera_z_i_yaw(),
            this->converter->get_img_t(),
            base::get_param<double>("auto-aim.top-model.single-angle-r"));
        this->counter.update_ab(
            length_sample.center,
            length_sample.ab,
            this->converter->get_img_t());
    } else {
        double z_to_l_fit = this->double_angle_sampling(
            armors,
            /*z_to_l_exp=*/
            aimer::math::reduced_angle(
                this->counter.predict_theta_l(this->converter->get_img_t())
                - this->converter->get_camera_z_i_yaw()),
            this->converter);

        aimer::debug::flask_aim << fmt::format(
            "zn_to_l[W]: {:.2f}",
            aimer::math::reduced_angle(z_to_l_fit + M_PI) / M_PI * 180.);

        this->counter.update_theta_l(
            this->converter->get_camera_z_i_yaw() + z_to_l_fit,
            this->converter->get_img_t(),
            base::get_param<double>("auto-aim.top-model.double-angle-r"));
    }
    // 无论如何都要估计角度，更新中心
    double z_to_l_fix = aimer::math::reduced_angle(
        this->counter.predict_theta_l(this->converter->get_img_t())
        - this->converter->get_camera_z_i_yaw());
    // 选取角度较小的装甲板（因为它的 pos 较精确）反向延长得到中心
    Eigen::Vector3d center_pro = (z_to_l_fix > M_PI / 4. * 3.)
        ? ([&]() { // 懒得传参数
              Eigen::Vector2d l_norm2 =
                  aimer::math::rotate(this->converter->get_camera_z_i2(), z_to_l_fix);
              // fix !
              int l_index = this->counter.get_pattern() == top4::TopPattern::AB ? 0 : 1;
              return top::prolonged_center(
                  armors[0].info.pos,
                  this->counter.get_ab(l_index).length,
                  l_norm2,
                  this->counter.get_ab(l_index).z_plus);
          }())
        : ([&]() {
              Eigen::Vector2d r_norm2 =
                  aimer::math::rotate(this->converter->get_camera_z_i2(), z_to_l_fix + M_PI / 2.);
              int r_index = this->counter.get_pattern() == top4::TopPattern::AB ? 1 : 0;
              return top::prolonged_center(
                  armors[1].info.pos,
                  this->counter.get_ab(r_index).length,
                  r_norm2,
                  this->counter.get_ab(r_index).z_plus);
          }());
    this->center_filter.update(
        center_pro,
        this->converter->get_img_t(),
        { 0.01, 10. },
        { base::get_param<double>("auto-aim.top-model.center-r-at-1m") * center_pro.norm()
          * center_pro.norm() });

    {
        // double 绘制反向延长中心
        auto cpt = this->converter->pi_to_pu(center_pro);
        aimer::debug::flask_aim << aimer::debug::FlaskPoint(cpt, { 0, 255, 255 }, 6, 3);
    }
}

void TopModel::update_by_single_armor() {
    const aimer::ArmorData& armor = this->state->get_armor_data_ref()[0];
    double z_to_l_exp = aimer::math::reduced_angle(
        this->counter.predict_theta_l(this->converter->get_img_t())
        - this->converter->get_camera_z_i_yaw());
    double z_to_l_fit =
        top::fit_single_z_to_v(armor, z_to_l_exp, M_PI / 4 * 3, M_PI / 4 * 5, this->converter);

    aimer::debug::flask_aim << fmt::format(
        "zn_to_l[S]: {:.2f}",
        aimer::math::reduced_angle(z_to_l_fit + M_PI) / M_PI * 180.);

    this->counter.update_theta_l(
        z_to_l_fit + this->converter->get_camera_z_i_yaw(),
        this->converter->get_img_t(),
        base::get_param<double>("auto-aim.top-model.single-angle-r"));
    double z_to_l_fix = aimer::math::reduced_angle(
        this->counter.predict_theta_l(this->converter->get_img_t())
        - this->converter->get_camera_z_i_yaw());
    Eigen::Vector3d center_pro = [&]() {
        int l_index = this->counter.get_pattern() == top4::TopPattern::A ? 0 : 1;
        Eigen::Vector2d l_norm2 =
            aimer::math::rotate(this->converter->get_camera_z_i2(), z_to_l_fix);
        return top::prolonged_center(
            armor.info.pos,
            this->counter.get_ab(l_index).length,
            l_norm2,
            this->counter.get_ab(l_index).z_plus);
    }();
    this->center_filter.update(
        center_pro,
        this->converter->get_img_t(),
        { 0.01, 10. },
        { base::get_param<double>("auto-aim.top-model.center-r-at-1m") * center_pro.norm()
          * center_pro.norm() });
}

void TopModel::draw_aim(cv::Mat& img) const {
    // 获取此刻结果
    std::vector<TopArmor> res =
        this->predict_armors(this->converter->get_img_t()); // 包含 Pattern 的具体实现
    // std::vector<TopArmor> res =
    // this->predict_armors(this->converter->get_prediction_time(
    //     this->converter->filter_to_aim_ypd(this->center_filter)
    // .dis));  // 包含 Pattern 的具体实现
    top::top_draw_aim(img, this->center_filter, res, this->active(), this->converter, this->state);
    // 绘制中心
    // converter->re_project_point(img, center_aim, {255, 87, 250});
}
} // namespace aimer::top::top4
