#include "aimer/auto_aim/predictor/aim/aim_corrector.hpp"

#include "aimer/auto_aim/predictor/aim/detect_bullet.hpp"
#include "aimer/auto_aim/predictor/aim/do_reproj.hpp"
#include "aimer/base/debug/debug.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"
#include "base/param/parameter.hpp"
#include "common/common.hpp"

namespace aimer::aim {

// modified in branch io_feature

// 类似于在 yml 中
const std::size_t AIM_CORRECTOR_AIM_HISTORY_MAX_SZ = 200u;
const std::size_t AIM_CORRECTOR_BULLETS_MAX_SZ = 200u;
const std::size_t AIM_CORRECTOR_PENDING_IDS_MAX_SZ = 200u;
const double AIM_CORRECTOR_AIM_TIME_MAX_ERROR = 50e-3;
const std::size_t AIM_CORRECTOR_ERROR_ANGLES_MAX_SZ = 15u;

// * 预估子弹捕获周围被识别子弹的距离除以半径的最大倍率。
// * 大于该倍率不会被捕获
const double CATCH_CIRCLE_DIS_MAX_RATIO = 4.;
const double CATCH_CIRCLE_RADIUS_MIN_RATIO = 1. / std::sqrt(2);
const double FIT_CIRCLE_MAX_T = 1.;
// * 15 次二分，精度为
const int FIT_CIRCLE_ITERATIONS_NUM = 15;

/** @class AimHistory */

// if a signature has no prefix, you can find its declaration
// in the closest named scope.

AimHistory::AimHistory(const std::size_t& max_sz): max_sz { max_sz } {}

// 插入一个包含 id, img_t 和 aim
auto AimHistory::add_aim(const aim::IdTLatencyAimCorrection& aim) -> void {
    // 正常情况下，输入的 id 递增。如果出现非递增，只能清空队列后才准确。
    if (this->aims.size() >= this->max_sz && !this->aims.empty()) {
        this->aims.pop_front();
    }
    if (this->aims.size() + 1u <= this->max_sz) {
        this->id_cnt += 1;
        this->aims.push_back(aim);
    }
}

auto AimHistory::get_id_cnt() const -> int {
    return this->id_cnt;
}

auto AimHistory::find_by_img_t(const double& img_t) const -> aim::IdTLatencyAimCorrection {
    // 比如发射延迟有 3s，但是只要发射时的枪口朝向能打到人就行
    auto it = std::lower_bound(
        this->aims.begin(),
        this->aims.end(),
        img_t,
        // lower_bound: 第一个违背
        [](const aim::IdTLatencyAimCorrection& a, const double& t) { return a.img_t < t; }
    );
    if (it == this->aims.end()) {
        return aim::IdTLatencyAimCorrection::invalid();
    }
    return *it;
}

auto AimHistory::find_by_id(const int& id) const -> aim::IdTLatencyAimCorrection {
    auto it = std::lower_bound(
        this->aims.begin(),
        this->aims.end(),
        id,
        [](const aim::IdTLatencyAimCorrection& a, const int& id) { return a.id < id; }
    );
    if (it == this->aims.end() || it->id != id) {
        return aim::IdTLatencyAimCorrection::invalid();
    }
    return *it;
}

/** @class ProjectileSimulator */
auto ProjectileSimulator::get_aim_ref() const -> const aim::IdTLatencyAimCorrection& {
    return this->aim;
}

auto ProjectileSimulator::get_fire_t() const -> double {
    return this->fire_t;
}

auto ProjectileSimulator::get_pos_by_t(const double& t) const -> aim::HitPos {
    // w, h 的启动点是枪口 barrel_target_pos
    // camera_target_pos -- 位移 -->
    // barrel_target_pos -- 抛物线补偿  -->
    // barrel_aim_pos
    const aimer::ShootParam shoot_param = this->aim.aim.shoot_param;
    double k = this->get_param_k();
    double w = (t - this->fire_t) * shoot_param.v0 * std::cos(shoot_param.aim_angle);
    // 高度为微分方程结果
    double h = (k * shoot_param.v0 * std::sin(shoot_param.aim_angle) + this->g) * k * w
            / (k * k * shoot_param.v0 * std::cos(shoot_param.aim_angle))
        + this->g * std::log(1. - (k * w) / (shoot_param.v0 * std::cos(shoot_param.aim_angle))) / k
            / k;
    // 弹道轨迹仅取决于目标点
    const Eigen::Vector3d target_xyz_i_barrel =
        this->converter->xyz_i_camera_to_xyz_i_barrel(shoot_param.target_xyz_i_camera);
    const Eigen::Vector3d w_norm =
        Eigen::Vector3d(target_xyz_i_barrel(0, 0), target_xyz_i_barrel(1, 0), 0).normalized();
    const Eigen::Vector3d h_norm = { 0., 0., 1. };
    const Eigen::Vector3d bullet_xyz_i_barrel = w * w_norm + h * h_norm;
    const Eigen::Vector3d bullet_xyz_i_camera =
        this->converter->xyz_i_barrel_to_xyz_i_camera(bullet_xyz_i_barrel);
    const Eigen::Vector2d bullet_xy_i_barrel = { bullet_xyz_i_barrel(0, 0),
                                                 bullet_xyz_i_barrel(1, 0) };
    const Eigen::Vector2d target_xy_i_barrel = { target_xyz_i_barrel(0, 0),
                                                 target_xyz_i_barrel(1, 0) };
    return aim::HitPos { bullet_xy_i_barrel.norm() >= target_xy_i_barrel.norm(),
                         bullet_xyz_i_camera };
}

auto ProjectileSimulator::get_pos() const -> aim::HitPos {
    return this->get_pos_by_t(this->converter->get_img_t());
}

auto ProjectileSimulator::get_circle_by_t(const double& t) const -> aim::HitCircle {
    aim::HitPos bullet = this->get_pos_by_t(t);
    Eigen::Vector3d xyz_c = this->converter->pi_to_pc(bullet.pos);
    // 沿着正 y 轴与视角的叉积方向得到一个边缘坐标，以计算半径
    Eigen::Vector3d crossed = Eigen::Vector3d(0., 1., 0.).cross(xyz_c).normalized();
    Eigen::Vector3d edge_xyz_c =
        xyz_c + crossed * base::get_param<double>("launching-mechanism.bullet.radius");
    Eigen::Vector3d edge_xyz_i = this->converter->pc_to_pi(edge_xyz_c);
    cv::Point2f edge_xy_u = this->converter->pi_to_pu(edge_xyz_i);
    cv::Point2f center_xy_u = this->converter->pi_to_pu(bullet.pos);
    float radius = aimer::math::get_dis(edge_xy_u, center_xy_u);
    return aim::HitCircle { bullet.hit, aimer::math::CircleF(edge_xy_u, radius) };
}

auto ProjectileSimulator::get_circle() const -> aim::HitCircle {
    return this->get_circle_by_t(this->converter->get_img_t());
}

auto ProjectileSimulator::get_param_k() const -> double {
    return base::get_param<double>("launching-mechanism.bullet.resistance-k");
}

auto ProjectileSimulator::catch_circle(const aimer::math::CircleF& circle) const
    -> aim::CaughtCost {
    aim::HitCircle hit_circle = this->get_circle();
    // 以内部 circle 为基准，因为外部的识别半径不太可信
    double dis = aimer::math::get_dis(hit_circle.circle.center, circle.center);
    double r_ratio = aimer::math::get_ratio(hit_circle.circle.r, circle.r);
    return aim::CaughtCost { dis <= aim::CATCH_CIRCLE_DIS_MAX_RATIO * hit_circle.circle.r
                                 && r_ratio >= aim::CATCH_CIRCLE_RADIUS_MIN_RATIO,
                             dis / (aim::CATCH_CIRCLE_DIS_MAX_RATIO * hit_circle.circle.r)
                                 + (1. - r_ratio) / (1. - aim::CATCH_CIRCLE_RADIUS_MIN_RATIO) };
}

auto ProjectileSimulator::fit_circle(const aimer::math::CircleF& circle) const
    -> aimer::math::CircleF {
    // 写一个二分类还是手写二分
    auto cost_func = [&](const double& t) {
        aimer::math::CircleF circle_in = this->get_circle_by_t(t).circle;
        // 时间越近越大
        return circle.r - circle_in.r;
    };
    aimer::math::Bisection solver;
    double t = solver
                   .find(
                       this->fire_t,
                       this->fire_t + aim::FIT_CIRCLE_MAX_T,
                       cost_func,
                       aim::FIT_CIRCLE_ITERATIONS_NUM
                   )
                   .first;
    return this->get_circle_by_t(t).circle;
}

/** @class AimCorrector */

// AimHistory 中存储了最近若干次可能发射的信息
// 模拟器 / 电控 会给出上一次发射的子弹，我们需要检查

AimCorrector::AimCorrector(aimer::CoordConverter* const converter):
    converter { converter },
    bullet_detector(aim::DoReproj(
        this->converter->get_f_cv_mat_ref(),
        this->converter->get_rot_ic_sup_cv_mat_ref()
    )),
    aim_history { aim::AIM_CORRECTOR_AIM_HISTORY_MAX_SZ } {
    for (auto& error_filter: this->error_filters) {
        error_filter.init_x(Eigen::Matrix<double, 1, 1> { 0. });
    }
}

auto AimCorrector::add_aim(const aim::IdTLatencyAimCorrection& aim) -> void {
    this->aim_history.add_aim(aim);
}

auto AimCorrector::update_bullet_id(const int& last_shoot_id) -> void {
    if (last_shoot_id != this->last_shoot_id) {
        this->last_shoot_id = last_shoot_id;
        if (this->pending_ids.size() + 1u <= aim::AIM_CORRECTOR_PENDING_IDS_MAX_SZ) {
            this->pending_ids.push(last_shoot_id);
        }
    }

    while (true) {
        // 如果 break 条件不止一个，把其中一个写在 while 的括号里可能带来误解
        if (this->pending_ids.empty()) {
            break;
        }
        int id = this->pending_ids.front();
        // 在 aim_history 列表中获取这个 id 对应的瞄准信息。
        aim::IdTLatencyAimCorrection origin_aim = this->aim_history.find_by_id(id);
        // 找不到 id 记录（已被删除），pop
        if (origin_aim.id == aim::IdTLatencyAimCorrection::INVALID_ID) {
            this->pending_ids.pop();
            continue;
        }
        // 设想 img_to_control = 0.5s
        // control_to_fire = 3s
        // 开火的 param 确实很晚才知道，而电控的 aim_id 也早就变成下一个了
        // 查询开火时的朝向对应的控制命令时间
        // 我们假定
        // 拥有该 img_t 的 aim 的实际执行时间为 img_t + img_to_control_latency
        double fire_controlling_t =
            origin_aim.img_t + this->converter->get_control_to_fire_latency();
        // + this->converter->get_img_to_fire_latency() -
        //     this->converter->get_img_to_control_latency();
        // 寻找 fire_controlling_t 后的第一个 aim 记录
        aim::IdTLatencyAimCorrection fire_controlling_aim =
            this->aim_history.find_by_img_t(fire_controlling_t);
        // id 确实在列表中，但所有记录 aim 的 t 均小于 id_fire_control_t，
        // 即该 id 发射时间 id_fire_t 的目标点尚未被计算，则所有 id 都需要等待
        if (fire_controlling_aim.id == aim::IdTLatencyAimCorrection::INVALID_ID) {
            break;
        }
        //            2.5(found) 3.5 4.5
        // id_fire_control_t = 1.5
        // 记录中最早的超过 t 的时间已经过晚，删除 find_t 并继续
        if (fire_controlling_t + aim::AIM_CORRECTOR_AIM_TIME_MAX_ERROR < fire_controlling_aim.img_t)
        {
            this->pending_ids.pop();
            continue;
        }
        if (this->bullets.size() + 1u <= aim::AIM_CORRECTOR_BULLETS_MAX_SZ) {
            // 发射朝向参数也许有点误差，但是发射时间最好无误差
            this->bullets.push_back(aim::IdProj {
                origin_aim.id,
                aim::ProjectileSimulator(
                    this->converter,
                    fire_controlling_aim,
                    // 发射朝向不是最理想估计，但时间是最理想估计
                    origin_aim.img_t + origin_aim.img_to_predict_latency
                        + this->converter->get_predict_to_send_latency()
                        + this->converter->get_send_to_control_latency()
                        + this->converter->get_control_to_fire_latency()
                ) });
        }
        this->pending_ids.pop();
        continue;
    }
}

auto AimCorrector::get_bullets() -> std::vector<aim::IdPos> {
    // 对尚未加入模拟的 id，寻找可能合适的模拟发射参数
    // 队列是一种尝试
    std::vector<aim::IdPos> res;
    for (auto it = this->bullets.begin(); it != this->bullets.end();) {
        if (this->converter->get_img_t() < it->proj.get_fire_t()) {
            ++it;
            continue;
        }
        aim::HitPos hit_pos = it->proj.get_pos_by_t(this->converter->get_img_t());
        // 已经击中了就可以删除了
        if (hit_pos.hit) {
            it = this->bullets.erase(it);
        } else {
            res.push_back(aim::IdPos { it->id, hit_pos.pos });
            ++it;
        }
    }
    return res;
}

auto AimCorrector::get_circles() -> std::vector<aim::IdCircle> {
    std::vector<aim::IdCircle> res;

    for (auto it = this->bullets.begin(); it != this->bullets.end();) {
        if (this->converter->get_img_t() < it->proj.get_fire_t()) {
            ++it;
            continue;
        }
        aim::HitCircle hit_circle = it->proj.get_circle_by_t(this->converter->get_img_t());
        if (hit_circle.hit) {
            it = this->bullets.erase(it);
        } else {
            res.push_back(aim::IdCircle { it->id, hit_circle.circle });
            ++it;
        }
    }
    return res;
}

auto AimCorrector::sample_aim_errors() -> void {
    aimer::debug::process_timer.print_process_time("before process");
    this->bullet_detector.process_new_frame(
        this->converter->get_img_ref(),
        this->converter->get_q()
    );
    aimer::debug::process_timer.print_process_time("after process");

    std::vector<aim::ImageBullet> detected = this->bullet_detector.bullets;
    // std::vector<aimer::math::CircleF> detected =
    //     this->bullet_detector.print_bullets();
    std::deque<aimer::math::CircleF> undistorted_detected;
    for (const auto& d: detected) {
        auto u = this->undistorted_circle(aimer::math::CircleF { d.center, d.radius });
        undistorted_detected.push_back(this->undistorted_circle(u));
        // aimer::debug::flask_aim << aimer::debug::FlaskPoint(d.center, {255, 0,
        // 255}, d.r,
        //                                               3);
        aimer::debug::flask_aim << aimer::debug::FlaskPoint(u.center, { 0, 255, 255 }, u.r, 3);
    }
    for (const auto& bullet: this->bullets) {
        // 寻找合法的目标中代价最小的
        // 此处可删除生成距离过远的
        auto best_bullet = undistorted_detected.end();
        aim::CaughtCost best_caught { false, 0. };
        for (auto it = undistorted_detected.begin(); it != undistorted_detected.end(); ++it) {
            aim::CaughtCost caught_cost = bullet.proj.catch_circle(*it);
            if (caught_cost.caught
                && (static_cast<int>(caught_cost.caught) > static_cast<int>(best_caught.caught)
                    || caught_cost.cost < best_caught.cost))
            {
                best_bullet = it;
                best_caught = caught_cost;
            }
        }
        if (best_bullet != undistorted_detected.end()) {
            // 根据大小二分得到校正的预估位置
            aimer::math::CircleF fit = bullet.proj.fit_circle(*best_bullet);
            aimer::debug::flask_aim
                << aimer::debug::FlaskPoint(fit.center, { 255, 255, 255 }, fit.r, 3);
            // 理想情况（枪口指向符合视觉预期）下的朝向角，相机球面坐标系，yaw
            // 正右，pitch 正下
            aimer::math::YpdCoord fit_yp = this->converter->pu_to_yp_c(fit.center);
            aimer::math::YpdCoord caught_yp = this->converter->pu_to_yp_c(best_bullet->center);
            // 实际 - 预期（理想发射）
            Eigen::Vector2d yp_error = { caught_yp.yaw - fit_yp.yaw,
                                         caught_yp.pitch - fit_yp.pitch };
            {
                std::vector<double> r_vec = {
                    base::get_param<double>("auto-aim.aim-corrector.error.r")
                };
                const Eigen::Vector2d& correction = bullet.proj.get_aim_ref().correction;
                // 理想的校正后，yp_error = 0，而 aim_ref 中是之前的 error
                this->error_filters[0].update(yp_error[0] + correction(0, 0), 0., { 1. }, r_vec);
                this->error_filters[1].update(yp_error[1] + correction(1, 0), 0., { 1. }, r_vec);
            }

            if (this->error_angles.size() + 1u > aim::AIM_CORRECTOR_ERROR_ANGLES_MAX_SZ
                && !this->error_angles.empty())
            {
                this->error_angles.pop_front();
            }
            if (this->error_angles.size() + 1u <= aim::AIM_CORRECTOR_ERROR_ANGLES_MAX_SZ) {
                this->error_angles.push_back(yp_error);
            }
            undistorted_detected.erase(best_bullet);
        }
    }
    {
        int i = 0;
        for (auto& d: this->error_angles) {
            i += 1;
            aimer::debug::auto_aim_page()
                ->sub("aim_corrector_采样数据")
                .sub("error" + std::to_string(i))
                .get() = fmt::format(
                "{:.2f}, {:.2f}",
                aimer::math::rad_to_deg(d(0, 0)),
                aimer::math::rad_to_deg(d(1, 0))
            );
        }
    }
}

auto AimCorrector::get_aim_error() const -> Eigen::Vector2d {
    return Eigen::Vector2d(
        this->error_filters[0].predict(0.)(0, 0),
        this->error_filters[1].predict(0.)(0, 0)
    );
}

auto AimCorrector::undistorted_circle(const aimer::math::CircleF& circle) -> aimer::math::CircleF {
    std::vector<float> rs;
    cv::Point2f circle_u = this->converter->pd_to_pu(circle.center);
    const cv::Point2f di[4] = { { 1.f, 0.f }, { 0.f, 1.f }, { -1.f, 0.f }, { 0.f, -1.f } };
    for (auto k: di) {
        cv::Point2f edge = circle.center + k * circle.r;
        cv::Point2f edge_u = this->converter->pd_to_pu(edge);
        rs.push_back(aimer::math::get_dis(circle_u, edge_u));
    }
    return aimer::math::CircleF(circle_u, aimer::math::get_vec_mean(rs));
}

} // namespace aimer::aim
