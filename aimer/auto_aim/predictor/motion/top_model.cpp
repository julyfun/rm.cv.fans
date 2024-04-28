#include "aimer/auto_aim/predictor/motion/top_model.hpp"

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>

// "" root location: closest CMakeLists
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"

namespace aimer::top {

const int FIND_ANGLE_ITERATIONS = 12; // 三分法迭代次数 理想精度 < 1.
const double SIMPLE_TOP_TRACK_AREA_RATIO = 2.;
const double DETECTOR_ERROR_PIXEL_BY_SLOPE = 2.;

double get_pts_cost(
    const std::vector<cv::point2f>& cv_refs,
    const std::vector<cv::point2f>& cv_pts,
    const double& inclined
) {
    std::size_t size = cv_refs.size();
    std::vector<eigen::vector2d> refs;
    std::vector<eigen::vector2d> pts;
    for (std::size_t i = 0u; i < size; ++i) {
        refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
        pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
    }
    double cost = 0.;
    for (std::size_t i = 0u; i < size; ++i) {
        std::size_t p = (i + 1u) % size;
        // i - p 构成线段。过程：先移动起点，再补长度，再旋转
        eigen::vector2d ref_d = refs[p] - refs[i]; // 标准
        Eigen::Vector2d pt_d = pts[p] - pts[i];
        // 长度差代价 + 起点差代价(1 / 2)（0 度左右应该抛弃)
        double pixel_dis = // dis 是指方差平面内到原点的距离
            (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm())
             + std::fabs(ref_d.norm() - pt_d.norm()))
            / ref_d.norm();
        double angular_dis = ref_d.norm() * aimer::math::get_abs_angle(ref_d, pt_d) / ref_d.norm();
        // 平方可能是为了配合 sin 和 cos
        // 弧度差代价（0 度左右占比应该大）
        double cost_i = math::sq(pixel_dis * std::sin(inclined))
            + math::sq(angular_dis * std::cos(inclined)) * top::DETECTOR_ERROR_PIXEL_BY_SLOPE;
        // 重投影像素误差越大，越相信斜率
        cost += std::sqrt(cost_i);
    }
    return cost;
}

std::vector<Eigen::Vector3d> radial_armor_corners(
    const Eigen::Vector3d& pos,
    const aimer::ArmorType& type,
    const double& pitch,
    const double& z_to_v,
    aimer::CoordConverter* const converter
) {
    const std::vector<cv::Point3d>& pw =
        type == aimer::ArmorType::BIG ? aimer::PW_BIG : aimer::PW_SMALL;
    Eigen::Vector2d radius_norm = aimer::math::rotate(converter->get_camera_z_i2(), z_to_v);
    Eigen::Vector3d x_norm;
    x_norm << aimer::math::rotate(radius_norm, M_PI / 2.), 0.;
    Eigen::Vector3d w_z_norm { 0., 0., 1. }; // 不要把变量写的像函数
    Eigen::Vector3d y_norm;
    y_norm << -radius_norm * std::sin(pitch), 0.;
    y_norm += w_z_norm * std::cos(pitch);
    std::vector<Eigen::Vector3d> corners;
    for (int i = 0; i < 4; ++i) {
        corners.push_back(pos + x_norm * pw[i].x + y_norm * pw[i].y);
    }
    return corners;
}

std::vector<cv::Point2f> radial_armor_pts(
    const Eigen::Vector3d& pos,
    const aimer::ArmorType& type,
    const double& pitch,
    const double& z_to_v,
    aimer::CoordConverter* const converter
) {
    std::vector<Eigen::Vector3d> corners =
        top::radial_armor_corners(pos, type, pitch, z_to_v, converter);
    std::vector<cv::Point2f> pts;
    for (int i = 0; i < 4; ++i) {
        pts.push_back(converter->pi_to_pu(corners[i]));
    }
    return pts;
}

std::vector<std::vector<cv::Point2f>> radial_double_pts(
    const std::vector<aimer::ArmorData>& armors,
    const double& z_to_l,
    aimer::CoordConverter* const converter
) {
    std::vector<std::vector<cv::Point2f>> res;
    res.push_back(top::radial_armor_pts(
        armors[0].info.pos,
        armors[0].info.sample.type,
        armors[0].info.orientation_pitch_under_rule,
        z_to_l,
        converter
    ));
    res.push_back(top::radial_armor_pts(
        armors[1].info.pos,
        armors[1].info.sample.type,
        armors[0].info.orientation_pitch_under_rule,
        z_to_l + M_PI / 2.,
        converter
    ));
    return res;
}

double SingleCost::operator()(const double& x) {
    // value: z_to_l
    std::vector<cv::Point2f> pts = top::radial_armor_pts(
        this->data.d.info.pos,
        this->data.d.info.sample.type,
        this->data.d.info.orientation_pitch_under_rule,
        x,
        this->data.converter
    );
    return top::get_pts_cost(
        pts,
        std::vector<cv::Point2f> {
            this->data.d.info.pus[0],
            this->data.d.info.pus[1],
            this->data.d.info.pus[2],
            this->data.d.info.pus[3],
        },
        this->data.z_to_v_exp
    );
}

double DoubleCost::operator()(const double& x) {
    double z_to_r_exp = this->data.z_to_l_exp + M_PI / 2.;
    std::vector<std::vector<cv::Point2f>> ml_pts =
        top::radial_double_pts(this->data.armors, x, this->data.converter);
    return top::get_pts_cost(
               ml_pts[0],
               std::vector<cv::Point2f> {
                   this->data.armors[0].info.pus[0],
                   this->data.armors[0].info.pus[1],
                   this->data.armors[0].info.pus[2],
                   this->data.armors[0].info.pus[3],
               },
               this->data.z_to_l_exp
           )
        + top::get_pts_cost(
               ml_pts[1],
               std::vector<cv::Point2f> {
                   this->data.armors[1].info.pus[0],
                   this->data.armors[1].info.pus[1],
                   this->data.armors[1].info.pus[2],
                   this->data.armors[1].info.pus[3],
               },
               z_to_r_exp
        );
}

double fit_single_z_to_v(
    const aimer::ArmorData& armor,
    const double& z_to_v_exp,
    const double& z_to_v_min,
    const double& z_to_v_max,
    aimer::CoordConverter* const converter
) {
    top::SingleCost single_cost = top::SingleCost(top::SingleData(armor, z_to_v_exp, converter));
    aimer::math::Trisection solver;
    std::pair<double, double> res =
        solver.find(z_to_v_min, z_to_v_max, single_cost, top::FIND_ANGLE_ITERATIONS);
    return aimer::math::reduced_angle(res.first);
    // double res = (z_to_v_min + z_to_v_max) / 2.;
    // ceres::Problem problem;
    // ceres::CostFunction* cost_function{
    //     new ceres::AutoDiffCostFunction<top::SingleCostFunctor, 1, 1>(
    //         new top::SingleCostFunctor(
    //             top::SingleData(armor, z_to_v_exp, converter)))};
    // problem.AddResidualBlock(cost_function, nullptr, &res);
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.minimizer_progress_to_stdout = true;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // return aimer::math::reduced_angle(res);
}

// 拟合得到角度。数据，限制，数据库
double fit_double_z_to_l(
    const std::vector<aimer::ArmorData>& armors,
    const double& z_to_l_exp,
    const double& z_to_l_min,
    const double& z_to_l_max,
    aimer::CoordConverter* const converter
) {
    top::DoubleCost double_cost { top::DoubleCost(top::DoubleData(armors, z_to_l_exp, converter)) };
    aimer::math::Trisection solver;
    std::pair<double, double> res =
        solver.find(z_to_l_min, z_to_l_max, double_cost, top::FIND_ANGLE_ITERATIONS);
    return aimer::math::reduced_angle(res.first);
}

// 不要为了规范而随便改旧的风格
// 1. 不知道旧的风格会不会被重新启用，主要看可读性
// 2. 提供旧风格与新风格的比较

aimer::AimInfo get_top_limit_aim(
    const top::CenterFilter& center_filter,
    const std::vector<top::TopArmor>& prediction_results,
    const std::vector<top::TopArmor>& hit_results,
    const double& top_w,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state
) {
    /// 获取瞄准点思路：
    /// 非常复杂...
    struct AimArmor {
        aimer::AimInfo emerging_aim;
        top::TopArmor armor;
    };
    auto get_direction = [](const double& top_w) -> top::TopDirection {
        return top_w > 0. ? top::TopDirection::CCW : top::TopDirection::CW;
    };

    auto get_directs = [](const std::vector<top::TopArmor>& results) -> std::vector<top::TopArmor> {
        std::vector<top::TopArmor> directs;
        for (const auto& res: results) {
            if (std::fabs(res.zn_to_v) < aimer::math::deg_to_rad(
                    base::get_param<double>("auto-aim.top-model.aim.max-orientation-angle")
                ))
            {
                directs.push_back(res);
            }
        }
        return directs;
    };

    auto choose_direct_aim = [&converter, &center_filter, &top_w, &get_direction](
                                 const std::vector<top::TopArmor>& directs
                             ) -> AimArmor {
        top::TopArmor direct = directs[0];
        for (const auto& d: directs) {
            // aim_swing_cost 利用的是与图像绑定的陀螺仪（枪口指向）信息选择打击板
            if (converter->aim_swing_cost(converter->target_pos_to_aim_ypd(d.pos))
                < converter->aim_swing_cost(converter->target_pos_to_aim_ypd(direct.pos)))
            {
                direct = d;
            }
        }
        aimer::AimInfo aim = [&]() {
            // 此处求解空气阻力函数复用
            aimer::ShootParam shoot_param = converter->target_pos_to_shoot_param(direct.pos);
            aimer::math::YpdCoord ypd =
                converter->aim_xyz_i_to_aim_ypd(shoot_param.aim_xyz_i_barrel);
            // 计算 yaw_v
            // 计算方法为“预测击中时敌人自角度”下根据“此刻角速度”推算 xy
            // 速度，然后 ypd 速度
            Eigen::Vector2d zn_norm2 = aimer::math::rotate(converter->get_camera_z_i2(), M_PI);
            // v_norm 是装甲板向量
            Eigen::Vector2d v_norm2 = aimer::math::rotate(zn_norm2, direct.zn_to_v);
            // xy 方向上的速度
            Eigen::Vector2d xy_v =
                aimer::math::rotate(
                    v_norm2,
                    get_direction(top_w) == top::TopDirection::CCW ? +M_PI / 2. // 垂直量
                                                                   : -M_PI / 2.
                )
                * direct.length * std::fabs(top_w);
            Eigen::Vector3d xyz_v = { xy_v(0, 0), xy_v(1, 0), 0. };
            aimer::math::YpdCoord ypd_v = converter->filter_to_aim_ypd_v(center_filter)
                + converter->get_camera_ypd_v(direct.pos, xyz_v);
            return aimer::AimInfo(ypd, ypd_v, shoot_param, ::ShootMode::IDLE);
        }();
        return AimArmor { aim, direct };
    };

    // results 应该包含时间
    auto get_indirect_aim = [&converter, &state, &center_filter, &top_w, &get_direction](
                                const std::vector<top::TopArmor>& results
                            ) -> AimArmor {
        double zn_to_lim = get_direction(top_w) == top::TopDirection::CCW
            ? -base::get_param<double>("auto-aim.top-model.aim.max-orientation-angle") / 180. * M_PI
            : +base::get_param<double>("auto-aim.top-model.aim.max-orientation-angle") / 180.
                * M_PI;
        // 所有装甲板距 limit 角的角度中最小的
        double closest_to_lim = aimer::INF;
        top::TopArmor indirect = results[0];
        for (const auto& armor: results) {
            // 跟踪限制最大角
            // 寻找即将出现的板子
            // 在新等待位置时允许超额打击（上一块板子）的角度，
            // 注意对于刚刚过去的板子，如果没有新目标就会傻傻等待并
            // 射击，但严格要求不跟踪（如英雄 0
            // 度)如果有新目标就会进入上方的 direct
            // 但当限制角很小的时候有可能打击这个刚刚过去的板子，此时
            // closest 是负值
            double leaving_angle =
                (state->get_sample_armor_ref().width / 2.
                 * base::get_param<double>("auto-aim.top-model.aim.max-out-error"))
                / armor.length;
            double armor_to_lim =
                aimer::math::reduced_angle(
                    (get_direction(top_w) == TopDirection::CCW ? zn_to_lim - armor.zn_to_v
                                                               : armor.zn_to_v - zn_to_lim)
                    - M_PI + leaving_angle
                )
                + M_PI - leaving_angle; // -leave ~ 2pi - leave
            if (armor_to_lim < closest_to_lim) {
                indirect = armor; // indirect
                closest_to_lim = armor_to_lim; // 允许负值出现，但是并不会跟踪
            }
        }
        // 匀速时理想 lim_center 固定
        // lim 度（例如 0）时的中心（欲瞄准
        // 用这个中心延长的位置）
        Eigen::Vector3d center_lim =
            center_filter.predict_pos(results[0].t + closest_to_lim / std::fabs(top_w));
        // what if top_w == 0?
        // 特例：当 lim = 0 且装甲板朝向已经转过去 5 度时，瞄准的依然是
        // 这块板在 5 度之前（正对）所在位置，此时 closest 的作用是
        // 防止枪口转到下一个装甲板正对时的位置
        Eigen::Vector2d center_lim2 = { center_lim(0, 0), center_lim(1, 0) };
        Eigen::Vector2d lim_norm2 =
            aimer::math::rotate(converter->get_camera_z_i2(), M_PI + zn_to_lim);
        Eigen::Vector2d emerging_pos2 = center_lim2 + lim_norm2 * indirect.length;
        Eigen::Vector3d emerging_pos = { emerging_pos2(0, 0),
                                         emerging_pos2(1, 0),
                                         center_lim(2, 0) + indirect.z_plus };
        aimer::AimInfo aim = [&]() {
            aimer::ShootParam shoot_param = converter->target_pos_to_shoot_param(emerging_pos);
            aimer::math::YpdCoord ypd =
                converter->aim_xyz_i_to_aim_ypd(shoot_param.aim_xyz_i_barrel);
            aimer::math::YpdCoord ypd_v = converter->filter_to_aim_ypd_v(center_filter);
            return aimer::AimInfo(ypd, ypd_v, shoot_param, ::ShootMode::IDLE);
        }();
        // 之前打击逻辑直接统一 在这里 indirect.pos
        return AimArmor { aim, indirect }; // 是根据 lim 延伸出来的
    };

    auto get_aim_angle = [&get_directs, &choose_direct_aim, &get_indirect_aim](
                             const std::vector<top::TopArmor>& results
                         ) -> AimArmor {
        std::vector<top::TopArmor> directs = get_directs(results);
        return !directs.empty() ? choose_direct_aim(directs) : get_indirect_aim(results);
    };

    aimer::AimInfo tracking_aim = get_aim_angle(prediction_results).emerging_aim;
    { // 以下判断 shoot
        // 常规 armor_model 的判断发弹为：当前是否足够收敛（理想 yaw 始终为 0）
        // 假设子弹为光速，那只要跟紧就根本不应该考虑发弹延迟，否则预估是错误的
        // 但实际上，就算当前不收敛，shoot cmd 延迟 = 3s 之
        // 后可能早已收敛，没办法预测自己在这么大延迟后的表现

        // 反陀螺则判断条件更少，因为更无法预测自己在极大 shoot cmd 延迟后是否收敛
        // 此时我们只能认为它保持收敛。判断发弹的方法更为基本，仅是该角度下被选定
        // 的板是否可能被打击
        // 但有一段时间（旋转到待击打点期间）不可能收敛

        // 若发射延迟达到装甲板切换间隔的一半，需要严肃考虑是否给发射指令
        // 此时给发射，击打位置在哪里？
        // img 的延迟也需要考虑。我们计算方法是时间轴上的击打时间点
        // 1. hit: 如果 hit
        // 位置需要旋转角太大，那就不打（主要是对于切换等待期间的限制）
        // 常规预测并无此判断，当发弹延迟巨大时，这一判断反而是累赘
        // 范围装甲板的打击限制）
        // 无法根据对 hit 收敛程度判断是否发射，hit 没有收敛之说 可否通过对
        AimArmor hit_aim = get_aim_angle(hit_results);
        bool for_a_hit = // 1. 电机有没有跟上
            converter->aim_error_exceeded(
                tracking_aim.ypd,
                state->get_sample_armor_ref(),
                state->get_aim_error(),
                hit_aim.armor.zn_to_v,
                state->get_armor_pitch()
            )
                // 2. 若打中，但将要打中的位置和当前枪口指向差的老远，就不打
                // 这玩意能删么
                // 不能删啊，跟随一块装甲板的最后一段时间是不能发发弹指令的
                // 改成 tracking 和 hitting 的差会不会更好？
                // 好像不行捏
                // 还是用来防止
                || converter->aim_error_exceeded(
                    hit_aim.emerging_aim.ypd,
                    state->get_sample_armor_ref(),
                    base::get_param<double>("auto-aim.top-model.aim.max-swing-error"),
                    hit_aim.armor.zn_to_v,
                    state->get_armor_pitch()
                )
                ||
                // 3. 这里函数是双 aim 比较的重载
                // 若打中，打中的 emerging pos 和实际装甲板位置的差，就是 out-error
                converter->aim_error_exceeded(
                    hit_aim.emerging_aim.ypd,
                    converter->target_pos_to_aim_ypd(hit_aim.armor.pos),
                    state->get_sample_armor_ref(),
                    base::get_param<double>("auto-aim.top-model.aim.max-out-error"),
                    hit_aim.armor.zn_to_v,
                    state->get_armor_pitch()
                )
            ? false
            : true;
        tracking_aim.shoot = for_a_hit ? ::ShootMode::SHOOT_NOW : ::ShootMode::TRACKING;
    }
    return tracking_aim;
}

// 绘制全车装甲板
void top_draw_aim(
    cv::Mat& img,
    const top::CenterFilter& center_filter,
    const std::vector<top::TopArmor>& top_results,
    bool top_active,
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state
) {
    Eigen::Vector3d center = center_filter.predict_pos(converter->get_img_t());
    aimer::debug::flask_map << aimer::debug::pos_str_to_map_text(
        fmt::format("{}", state->get_number()),
        center,
        aimer::debug::FLASK_MAP_TEXT_COLOR,
        aimer::debug::FLASK_MAP_TEXT_SCALE
    );

    std::vector<std::vector<cv::Point2f>> pts_vec;
    std::vector<std::vector<Eigen::Vector3d>> armors_corners;
    for (const auto& d: top_results) {
        pts_vec.push_back(top::radial_armor_pts(
            d.pos,
            state->get_sample_armor_ref().type,
            state->get_armor_pitch(),
            d.zn_to_v + M_PI,
            converter
        ));
        armors_corners.push_back(top::radial_armor_corners(
            d.pos,
            state->get_sample_armor_ref().type,
            state->get_armor_pitch(),
            d.zn_to_v + M_PI,
            converter
        ));
    }
    const int size = top_results.size();
    const int thickness = top_active ? 2 : 1;
    for (int i = 0; i < size; ++i) {
        int p = (i + 1) % size;
        double coe_i = 1. + 0. * (1. - std::fabs(top_results[i].zn_to_v) / M_PI);
        double coe_p = 1. + 0. * (1. - std::fabs(top_results[i].zn_to_v) / M_PI);
        cv::Scalar color_armor = { 0, 255, 0 };
        cv::Scalar color_sp = { 255, 87, 250 };
        aimer::debug::draw_lines(img, pts_vec[i], color_armor * coe_i, thickness, true);
        aimer::debug::draw_line(
            img,
            pts_vec[i][3],
            pts_vec[p][0],
            color_sp * ((coe_i + coe_p) / 2.),
            thickness
        );
        aimer::debug::draw_line(
            img,
            pts_vec[i][2],
            pts_vec[p][1],
            color_sp * ((coe_i + coe_p) / 2.),
            thickness
        );
        aimer::debug::flask_map << aimer::debug::poses_to_map_lines(
            armors_corners[i],
            state->is_hit() ? aimer::debug::GRAY_MAIN_COLOR : color_armor,
            true,
            aimer::debug::FLASK_MAP_THICKNESS
        );
        aimer::debug::flask_map << aimer::debug::poses_to_map_lines(
            { armors_corners[i][3], armors_corners[p][0] },
            state->is_hit() ? aimer::debug::GRAY_MAIN_COLOR : color_sp,
            false,
            aimer::debug::FLASK_MAP_THICKNESS
        );
        aimer::debug::flask_map << aimer::debug::poses_to_map_lines(
            { armors_corners[i][2], armors_corners[p][1] },
            state->is_hit() ? aimer::debug::GRAY_MAIN_COLOR : color_sp,
            false,
            aimer::debug::FLASK_MAP_THICKNESS
        );
    }
}

Eigen::Vector3d prolonged_center(
    const Eigen::Vector3d& armor_pos,
    const double& radius,
    const Eigen::Vector2d& radius_norm,
    const double& z_plus
) {
    Eigen::Vector3d center_pro;
    center_pro << armor_pos(0, 0) - radius * radius_norm(0, 0),
        armor_pos(1, 0) - radius * radius_norm(1, 0), armor_pos(2, 0) - z_plus;
    return center_pro;
}

/**
 * @class SimpleRotateCounter
 *
 */
SimpleRotateCounter::SimpleRotateCounter(
    const int& armor_cnt,
    const double& jump_angle,
    const double& min_active_rotate
):
    armor_cnt(armor_cnt),
    jump_angle(jump_angle),
    min_active_rotate(min_active_rotate) {}

int SimpleRotateCounter::get_active_rotate() const {
    return this->active_rotate;
}

int SimpleRotateCounter::get_rotate() const {
    return this->rotate;
}

double SimpleRotateCounter::get_w() const {
    return this->theta_s_filter.get_w();
}

double SimpleRotateCounter::predict_theta_v(const double& t) const {
    double theta_s = this->theta_s_filter.predict_angle(t);
    return aimer::math::reduced_angle(theta_s - this->rotate * (M_PI * 2. / this->armor_cnt));
}

bool SimpleRotateCounter::w_active() const { // 旋转状态
    return std::fabs(this->get_w())
        > base::get_param<double>("auto-aim.top-model.active-w") / 180. * M_PI;
}

bool SimpleRotateCounter::w_inactive() const {
    return std::fabs(this->get_w())
        <= base::get_param<double>("auto-aim.top-model.inactive-w") / 180. * M_PI;
}

bool SimpleRotateCounter::top_active() const {
    return !this->w_inactive() && std::fabs(this->get_active_rotate()) >= this->min_active_rotate;
}

void SimpleRotateCounter::update_theta_v(const double& theta_v, const double& t, const double& R) {
    this->theta_s_filter.update(theta_v + this->rotate * (M_PI * 2. / this->armor_cnt), t, R);
}

// 在 update_rotate 之前并不知道 predict_v 是否真的是 l
void SimpleRotateCounter::update_rotate(
    const aimer::ArmorData& d,
    const double& zn_to_v,
    const double& t
) {
    // 强修复
    // 当前帧区间
    bool top_pre_active = this->top_active();
    int seg = (std::fabs(zn_to_v) > this->jump_angle ? 1 : 0) * (zn_to_v > 0. ? 1 : -1);
    // 在边缘会出现严重失帧，因此该方法比直接判断相差角更稳定
    bool armor_changed = d.id != this->last_id && seg * this->last_seg == -1;
    if (armor_changed) {
        this->id_direction = seg < 0. ? TopDirection::CCW : TopDirection::CW;
        if (this->id_direction == TopDirection::CCW) {
            ++this->rotate;
            ++this->active_rotate;
        } else {
            --this->rotate;
            --this->active_rotate;
        }
    }
    if (top_pre_active) {
        if (this->w_inactive()) {
            this->active_rotate = 0;
        }
    } else {
        if (!this->w_active()) {
            this->active_rotate = 0;
        }
    }
    this->last_id = d.id;
    this->last_seg = seg;
}

void AngleReserver::init(const double& angle) {
    this->early = angle;
    this->late = angle;
}

void AngleReserver::update(const double& angle) {
    this->late = angle;
}

double AngleReserver::get_early() {
    return this->early;
}

double AngleReserver::get_late() {
    return this->late;
}

OrientationSignFixer::OrientationSignFixer(const OrientationSignFixerConstructor& cons):
    reset_time(cons.reset_time),
    sampling_time(cons.sampling_time),
    reserving_range(cons.reserving_range) {}
// 用上一周期的 yaw 修复 angle 的符号
double OrientationSignFixer::fixed(const double& angle, const double& yaw) const {
    if (std::fabs(angle) <= this->reserving_range) {
        return angle;
    }
    double credit_mid_yaw =
        aimer::math::get_weighted_angle(this->credit_min_yaw, 0.5, this->credit_max_yaw, 0.5);
    // 对镜头而言，向左为正，对板子来说，它向左则它在负数
    // 对镜头 mid to yaw 需要加，由于对较大为向左，故是负数
    if (aimer::math::get_rotate_angle(credit_mid_yaw, yaw) > 0.) {
        return -std::fabs(angle);
    }
    return std::fabs(angle);
}

void OrientationSignFixer::update(const double& yaw, const double& t) {
    if (t - this->init_t > this->reset_time) {
        this->reserver.init(yaw);
        this->init_t = t; // 负面的，新建周期
    }
    if (this->update_t - this->init_t > this->sampling_time) {
        // 本帧暂时不知道是不是拓展
        // 自身可信，回头即重置
        double early_to_late =
            aimer::math::get_rotate_angle(this->reserver.get_early(), this->reserver.get_late());
        double mid = aimer::math::get_weighted_angle(
            this->reserver.get_early(),
            0.5,
            this->reserver.get_late(),
            0.5
        );
        double mid_to_cur = aimer::math::get_rotate_angle(mid, yaw);
        if (mid_to_cur * early_to_late < 0.) {
            // 新周期，重置存储器
            // wtf!!!!
            this->credit_max_yaw =
                aimer::math::max_angle(this->reserver.get_early(), this->reserver.get_late());
            this->credit_min_yaw =
                aimer::math::min_angle(this->reserver.get_early(), this->reserver.get_late());
            this->reserver.init(yaw);
            this->init_t = t;
        }
    }
    // 方向正确后，进行拓展
    this->reserver.update(yaw);
    this->update_t = t;
}

SimpleTopModel::SimpleTopModel(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state,
    const int& armor_cnt,
    const double& jump_angle,
    const int& min_active_rotate,
    const double& credit_time,
    const OrientationSignFixerConstructor& cons
):
    converter(converter),
    state(state),
    armor_cnt(armor_cnt),
    credit_clock(converter, credit_time),
    counter(armor_cnt, jump_angle, min_active_rotate),
    orientation_sign_fixer(cons) {}

bool SimpleTopModel::active() const {
    return this->counter.top_active();
}

void SimpleTopModel::update(const bool& enable_sign_fixer) {
    const std::vector<aimer::ArmorData>& armors = this->state->get_armor_data_ref();
    // 更新失败
    if (armors.size() != 1 && armors.size() != 2) {
        return;
    }

    const bool credit_before_update = this->credit_clock.credit();

    this->credit_clock.update();
    aimer::ArmorData tracking_armor = armors[0]; // 设置默认值
    {
        // 尽可能持续追踪同一板，直到消失或者新的某一板面积远大于它
        // 以免反复横跳的问题
        double id_area = 0., max_id = -1, max_area = 0.;
        for (const auto& d: armors) {
            if (d.id == this->tracking_id) {
                id_area = d.info.area(); // maybe not found
                tracking_armor = d;
            }
            if (d.info.area() > max_area) {
                max_id = d.id;
                max_area = d.info.area();
            }
        }
        for (const auto& d: armors) {
            if (d.id == this->tracking_id) {
                continue;
            }
            if (d.id == max_id && d.info.area() > id_area * top::SIMPLE_TOP_TRACK_AREA_RATIO) {
                this->tracking_id = d.id;
                tracking_armor = d;
            }
        }
    }

    this->orientation_sign_fixer.update(
        aimer::math::xyz_to_ypd(tracking_armor.info.pos).yaw,
        this->converter->get_img_t()
    );
    // 此处期望角不可用 exp，当 exp 为 0 时始终非常相信斜率，导致 raw 始终估计为 0
    // 左右
    double zn_to_v_raw = [&]() {
        double zn_to_v = aimer::math::reduced_angle(
            top::fit_single_z_to_v(
                tracking_armor,
                M_PI / 4.,
                M_PI / 2.,
                M_PI / 2. * 3.,
                this->converter
            )
            - M_PI
        );
        // 有时正负颠倒，但不影响 fabs 判断
        if (enable_sign_fixer) {
            return this->orientation_sign_fixer.fixed(
                zn_to_v,
                aimer::math::xyz_to_ypd(tracking_armor.info.pos).yaw
            );
        }
        return zn_to_v;
    }();

    this->counter.update_rotate(tracking_armor, zn_to_v_raw, this->converter->get_img_t());

    // 帧率高时将严重拉低速度
    // 获取和更新角度
    {
        double z_to_v_exp = aimer::math::reduced_angle(
            this->counter.predict_theta_v(this->converter->get_img_t())
            - this->converter->get_camera_z_i_yaw()
        ); // reserve for debug
        double z_to_v_fit = aimer::math::reduced_angle([&]() {
            double zn_to_v = aimer::math::reduced_angle(
                top::fit_single_z_to_v(
                    tracking_armor,
                    z_to_v_exp,
                    M_PI / 2.,
                    M_PI / 2. * 3.,
                    this->converter
                )
                - M_PI
            );
            if (enable_sign_fixer) {
                return this->orientation_sign_fixer.fixed(
                    zn_to_v,
                    aimer::math::xyz_to_ypd(tracking_armor.info.pos).yaw
                );
            }
            return zn_to_v;
        }() + M_PI);

        this->counter.update_theta_v(
            z_to_v_fit + this->converter->get_camera_z_i_yaw(),
            this->converter->get_img_t(),
            base::get_param<double>("auto-aim.top-model.single-angle-r")
        );
        // }
    }
    // 计算和更新中心
    {
        // 不一定 update 过，所以不用 get
        // fix 之前必须有 update，不然就退化成 exp
        // 后验：posterior
        double z_to_v_fix = aimer::math::reduced_angle(
            this->counter.predict_theta_v(this->converter->get_img_t())
            - this->converter->get_camera_z_i_yaw()
        );
        Eigen::Vector2d l_norm2 =
            aimer::math::rotate(this->converter->get_camera_z_i2(), z_to_v_fix);
        Eigen::Vector3d center_pro = top::prolonged_center(
            tracking_armor.info.pos,
            this->state->get_default_radius(),
            l_norm2,
            0.
        );

        if (this->state->get_enemy_type() == aimer::EnemyType::OUTPOST) {
            const double center_update_max_angle = aimer::math::deg_to_rad(
                base::get_param<double>("auto-aim.top-model.outpost.center-update-max-angle")
            );
            const double zn_to_v_fix = aimer::math::reduced_angle(z_to_v_fix + M_PI);
            if (!credit_before_update) {
                this->center_filter.set_pos(center_pro);
            } else if (std::abs(zn_to_v_fix) <= center_update_max_angle) {
                this->center_filter.update(
                    center_pro,
                    this->converter->get_img_t(),
                    { 0.01, 10. },
                    { base::get_param<double>("auto-aim.top-model.center-r-at-1m")
                      * center_pro.norm() * center_pro.norm() }
                );
                this->center_filter.set_v({ 0.0, 0.0, 0.0 });
            } else {
                // do nothing...
            }
        } else {
            this->center_filter.update(
                center_pro,
                this->converter->get_img_t(),
                { 0.01, 10. },
                { base::get_param<double>("auto-aim.top-model.center-r-at-1m") * center_pro.norm()
                  * center_pro.norm() }
            );
        }
    }
}

std::vector<top::TopArmor> SimpleTopModel::predict_armors(const double& t) const {
    Eigen::Vector3d center = this->center_filter.predict_pos(t);
    double theta = this->counter.predict_theta_v(t);
    Eigen::Vector2d norm2 = { 1., 0. };
    norm2 = aimer::math::rotate(norm2, theta);
    double theta_zn = this->converter->get_camera_z_i_yaw() + M_PI;
    std::vector<top::TopArmor> res;
    for (int i = 0; i < this->armor_cnt; ++i) {
        double radius = this->state->get_default_radius();
        Eigen::Vector3d pos { center(0, 0) + radius * norm2(0, 0),
                              center(1, 0) + radius * norm2(1, 0),
                              center(2, 0) };
        res.emplace_back(pos, t, aimer::math::reduced_angle(theta - theta_zn), radius, 0.);
        norm2 = aimer::math::rotate(norm2, 2. * M_PI / this->armor_cnt);
        theta += 2. * M_PI / this->armor_cnt;
    }
    return res;
}

aimer::AimInfo SimpleTopModel::get_limit_aim() const {
    if (!this->credit_clock.credit()) {
        return aimer::AimInfo::idle();
    }
    std::vector<top::TopArmor> prediction_results =
        this->predict_armors(this->converter->filter_to_prediction_time(this->center_filter));
    std::vector<top::TopArmor> hit_results =
        this->predict_armors(this->converter->filter_to_hit_time(this->center_filter));
    return top::get_top_limit_aim(
        this->center_filter,
        prediction_results,
        hit_results,
        this->counter.get_w(),
        this->converter,
        this->state
    );
}

void SimpleTopModel::draw_aim(cv::Mat& img) const {
    std::vector<top::TopArmor> res = this->predict_armors(this->converter->get_img_t());
    // 获取延迟结果
    // std::vector<top::TopArmor> res = this->predict_armors(
    //     converter->get_img_t() +
    //     converter->get_fire_to_hit_latency(aim_center.dis), converter);
    top::top_draw_aim(img, this->center_filter, res, this->active(), this->converter, this->state);
}

} // namespace aimer::top
