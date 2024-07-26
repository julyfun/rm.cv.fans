#include "aimer/auto_aim/predictor/motion/armor_model.hpp"

#include "aimer/base/robot/coord_converter.hpp"

namespace aimer {
/** @struct FilterThread */
FilterThread::FilterThread(
    aimer::CoordConverter* const converter,
    const aimer::ArmorData& armor,
    const double& credit_time
):
    armor(armor),
    credit_clock(converter, credit_time) {
    // 确认更新即可 update clock
    this->credit_clock.update();
    this->filter.init(aimer::math::xyz_to_ypd(armor.info.pos), converter->get_img_t());
}

void FilterThread::update(const aimer::ArmorData& armor, const double& t) {
    this->credit_clock.update();
    this->armor = armor;
    aimer::math::YpdCoord ypd = aimer::math::xyz_to_ypd(armor.info.pos);
    // 已经改成 ekf
    std::vector<double> q_vec = []() {
        double q_x = base::get_param<double>("auto-aim.armor-model.ekf.q.x");
        double q_v = base::get_param<double>("auto-aim.armor-model.ekf.q.v");
        return std::vector<double> { q_x, q_v, q_x, q_v, q_x, q_v };
    }();
    std::vector<double> r_vec { base::get_param<double>("auto-aim.armor-model.ekf.r.yaw"),
                                base::get_param<double>("auto-aim.armor-model.ekf.r.pitch"),
                                base::get_param<double>("auto-aim.armor-model.ekf.r.distance-at-1m")
                                    * ypd.dis * ypd.dis };
    this->filter.update(ypd, t, q_vec, r_vec);
    // 该帧相机数据总不可能存于 aimer::ArmorData 中
}

bool FilterThread::credit() const {
    return this->credit_clock.credit();
}

/** @class MotionModel */
ArmorModel::ArmorModel(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state,
    const double& credit_time
):
    converter(converter),
    state(state),
    credit_time(credit_time) {}

void ArmorModel::draw_aim(cv::Mat& img) const {
    for (const auto& d: this->filters) {
        double prediction_t = this->converter->filter_to_prediction_time(d.second.filter);
        Eigen::Vector3d pos = d.second.filter.predict_pos(this->converter->get_img_t());
        Eigen::Vector3d v = d.second.filter.predict_v(prediction_t);
        cv::Scalar color = this->state->is_hit() ? aimer::debug::GRAY_MAIN_COLOR
            : this->converter->get_robot_status_ref().enemy_color == ::EnemyColor::BLUE
            ? aimer::debug::BLUE_MAIN_COLOR
            : aimer::debug::RED_MAIN_COLOR;
        aimer::debug::flask_map << aimer::debug::pos_to_map_point(
            pos,
            color,
            aimer::debug::FLASK_MAP_PT_RADIUS,
            aimer::debug::FLASK_MAP_THICKNESS
        );
        aimer::debug::flask_map << aimer::debug::pos_pair_to_map_arrow(
            std::make_pair(pos, pos + v * 1.),
            color,
            aimer::debug::FLASK_MAP_THICKNESS
        );
        aimer::debug::flask_map << aimer::debug::pos_str_to_map_text(
            fmt::format("{}", this->state->get_number()),
            pos,
            color,
            aimer::debug::FLASK_MAP_TEXT_SCALE / 2.
        );
    }
}

// 建立若干线程，取激活的进行（一旦消失，立即取消激活，最精确 - 帧相关）
// 本模型为帧相关
void ArmorModel::update() {
    const std::vector<aimer::ArmorData>& armors = this->state->get_armor_data_ref();
    // 无法更新
    if (armors.empty()) {
        return;
    }

    for (const auto& d: armors) {
        auto it = this->filters.find(d.id);
        if (it == this->filters.end()) {
            this->filters.insert(
                std::make_pair(d.id, FilterThread(this->converter, d, this->credit_time))
            );
            it = this->filters.find(d.id);
        } else {
            it->second.update(d, this->converter->get_img_t());
        }
        // never check by converter frame
        // 最多 update (active) 不超过 armors 数量
    }
    // state 中不应该有 converter，因为他们各自传给各个模型
    for (auto it = this->filters.begin(); it != this->filters.end();) {
        if (!it->second.credit()) {
            it = this->filters.erase(it);
        } else {
            ++it;
        }
    }

    // std::cout << "Motion model threads count: " << this->filters.size() <<
    // '\n';
}

// 根据线程死活进行判断是否 idle
aimer::AimInfo ArmorModel::get_aim(const bool& passive) const {
    if (passive) {
        return this->get_passive_aim();
    }
    return this->get_positive_aim();
}

aimer::AimInfo ArmorModel::get_positive_aim() const {
    double max_area = 0.;
    for (const auto& d: this->filters) {
        if (d.second.credit() && d.second.armor.info.area() > max_area) {
            max_area = d.second.armor.info.area();
        }
    }
    aimer::AimInfo aim = aimer::AimInfo::idle(); // IDLE 弱于 TRACKING 和 SHOOT_NOW
    for (const auto& d: this->filters) {
        if (!d.second.credit()
            || d.second.armor.info.area()
                < max_area * base::get_param<double>("auto-aim.armor-model.target-area-ratio"))
        {
            continue;
            // 面积小于最大 0.5 以下的不打，然后按照最小旋转角排序。
            // 当然不会优先选择上一帧的目标
        }
        // control 的时候它已经到哪里了，...
        aimer::AimInfo filter_aim = [&]() -> aimer::AimInfo {
            aimer::ShootParam shoot_param = this->converter->filter_to_shoot_param(d.second.filter);
            // Eigen::Vector3d target_pos =
            //     this->converter->filter_to_shoot_param(d.second.filter);
            // target_pos_to_aim_ypd 非常复杂，包含玄学修正
            aimer::math::YpdCoord ypd =
                this->converter->aim_xyz_i_to_aim_ypd(shoot_param.aim_xyz_i_barrel);
            aimer::math::YpdCoord ypd_v = this->converter->filter_to_aim_ypd_v(d.second.filter);
            return aimer::AimInfo(
                ypd,
                ypd_v,
                shoot_param,
                this->converter->aim_error_exceeded(
                    ypd,
                    this->state->get_sample_armor_ref(),
                    this->state->get_aim_error(),
                    0.,
                    this->state->get_armor_pitch()
                )
                    ? ::ShootMode::TRACKING
                    : ::ShootMode::SHOOT_NOW
            );
        }();
        if (this->converter->aim_cmp(filter_aim, aim)) {
            aim = filter_aim;
        }
    }
    return aim;
}

aimer::AimInfo ArmorModel::get_passive_aim() const {
    double max_area = 0.;
    for (const auto& d: this->filters) {
        if (d.second.credit() && d.second.armor.info.area() > max_area) {
            max_area = d.second.armor.info.area();
        }
    }
    aimer::AimInfo aim = aimer::AimInfo::idle(); // 选择最优目标
    for (const auto& d: this->filters) {
        if (!d.second.credit()
            || d.second.armor.info.area()
                < max_area * base::get_param<double>("auto-aim.armor-model.target-area-ratio"))
        {
            continue; // 面积小于最大 0.5 以下的不打，然后按照中心排序
        }
        aimer::AimInfo filter_aim = [&]() -> aimer::AimInfo {
            Eigen::Vector3d hit_pos = this->converter->filter_to_hit_pos(d.second.filter);
            aimer::ShootParam shoot_param = this->converter->target_pos_to_shoot_param(hit_pos);
            aimer::math::YpdCoord ypd =
                this->converter->aim_xyz_i_to_aim_ypd(shoot_param.aim_xyz_i_barrel);
            aimer::math::YpdCoord ypd_v = this->converter->filter_to_hit_aim_ypd_v(d.second.filter);
            return aimer::AimInfo(
                { this->converter->get_control_aim_yaw0(), ypd.pitch, ypd.dis },
                { 0., 0., ypd_v.dis },
                shoot_param,
                // 若仅考虑 yaw 就超误差，不跟随
                this->converter->aim_error_exceeded(
                    aimer::math::YpdCoord(
                        ypd.yaw,
                        this->converter->get_control_aim_pitch0(),
                        ypd.dis
                    ),
                    this->state->get_sample_armor_ref(),
                    base::get_param<double>("auto-aim.armor-model.passive-mode.aim.tracking-range"),
                    0.,
                    this->state->get_armor_pitch()
                )
                    ? ::ShootMode::IDLE
                    // 若 yaw 不超误差
                    // - 若考虑 yaw 或 pitch 超误差，仅跟随不发射
                    : this->converter->aim_error_exceeded(
                          ypd,
                          this->state->get_sample_armor_ref(),
                          base::get_param<double>("auto-aim.armor-model.passive-mode.aim.max-error"
                          ),
                          0.,
                          this->state->get_armor_pitch()
                      )
                    ? ::ShootMode::TRACKING
                    : ::ShootMode::SHOOT_NOW
            );
        }();
        // idle 会被替代
        if (this->converter->aim_cmp(filter_aim, aim)) {
            aim = filter_aim;
        }
    }
    return aim;
}
} // namespace aimer
