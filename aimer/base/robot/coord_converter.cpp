#include "aimer/base/robot/coord_converter.hpp"

#include <ceres/types.h>
#include <fmt/format.h>

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"
#include "base/param/parameter.hpp"

namespace aimer {

const double MIN_BULLET_SPEED = 10.; // m/s
const double MIN_IMG_TO_PREDICT_LATENCY = 0.001;
const double MAX_IMG_TO_PREDICT_LATENCY = 0.050;
// 测试中，显示为 13~17 ms
const double DEFAULT_IMG_TO_PREDICT_LATENCY = 0.015;

/**
 * @class ResistanceFuncLinear
 * @brief 含空气阻力的弹道计算
 */
class ResistanceFuncLinear {
private:
    const double g { 9.8 }; // g = 9.8
    const double w, h, v0;

    double get_param_k() const {
        return base::get_param<double>("launching-mechanism.bullet.resistance-k");
    }

public:
    ResistanceFuncLinear(const double& w, const double& h, const double& v0): w(w), h(h), v0(v0) {}

    template<typename T>
    bool operator()(const T* const x, T* residual) const {
        const double k = this->get_param_k();
        residual[0] = (k * this->v0 * ceres::sin(x[0]) + this->g) * k * this->w
                / (k * k * this->v0 * ceres::cos(x[0]))
            + this->g * ceres::log(1. - (k * this->w) / (this->v0 * ceres::cos(x[0]))) / k / k
            - this->h;
        return true;
    }
};

/** @class aimer::CoordConverter */

// 注意 pts 和 pus
// pc， pi 始终在相机系统下
// 仅对于 aim_ypd 改用枪管系统
// 但注意，img 和四元数唯一对应，看到反向拟合的应该是对的
CoordConverter::CoordConverter():
    robot_status_obj { umt::ObjManager<RobotStatus>::find_or_create("robot_status") } {
    cv::FileStorage fin(CMAKE_DEF_PROJECT_DIR "/assets/param.yml", cv::FileStorage::READ);
    fin["R_CI"] >> this->rot_ic_sup_cv_mat;
    fin["F"] >> this->f_cv_mat;
    fin["C"] >> this->c_cv_mat;
    cv::cv2eigen(this->rot_ic_sup_cv_mat, this->rot_ic_sup);
    cv::cv2eigen(this->f_cv_mat, this->f_mat);
    cv::cv2eigen(this->c_cv_mat, this->c_mat);

    {
        char buf2[64];
        std::time_t t;
        std::time(&t);
        std::tm* tm = std::localtime(&t);
        std::strftime(buf2, 64, "%Y-%m-%d-%H-%M-%S_coord_log.txt", tm);
        this->file_str = buf2;
    }
}

const cv::Mat& CoordConverter::get_img_ref() const {
    return this->img;
}

// 获取帧计数
int CoordConverter::get_frame() const {
    return this->frame;
}

// 输出图像对应的相机时间
double CoordConverter::get_img_t() const {
    return this->img_t;
}

const RobotStatus& CoordConverter::get_robot_status_ref() const {
    return *this->robot_status_obj;
}

double CoordConverter::get_yaw_compensate() const {
    return base::get_param<bool>("ec.using-angle-compensate")
        ? this->get_robot_status_ref().yaw_compensate / 180. * M_PI
        : 0.;
}

double CoordConverter::get_pitch_compensate() const {
    return base::get_param<bool>("ec.using-angle-compensate")
        ? this->get_robot_status_ref().pitch_compensate / 180. * M_PI
        : 0.;
}

double CoordConverter::get_bullet_speed() const {
    return std::max(double(this->get_robot_status_ref().bullet_speed), aimer::MIN_BULLET_SPEED);
}

double CoordConverter::get_img_to_predict_latency() const {
    double latency = this->predict_timestamp_binder.get(this->get_frame()) - this->img_t;
    return aimer::math::clamp_default(
        latency,
        aimer::MIN_IMG_TO_PREDICT_LATENCY,
        aimer::MAX_IMG_TO_PREDICT_LATENCY,
        aimer::DEFAULT_IMG_TO_PREDICT_LATENCY
    );
}

double CoordConverter::get_predict_to_send_latency() const { // estimated
    return this->predict_to_send_latency_filter.predict(this->get_img_t())(0, 0);
}

double CoordConverter::get_send_to_control_latency() const {
    return base::get_param<double>("auto-aim.latency.send-to-control");
}

double CoordConverter::get_control_to_fire_latency() const {
    // double latency =
    //     double(int(this->get_robot_status().latency_cmd_to_fire)) / 1e3;
    // return aimer::math::in_range(
    //     latency, aimer::MIN_CONTROL_TO_FIRE_LATENCY,
    //     aimer::MAX_CONTROL_TO_FIRE_LATENCY,
    //     base::get_param<double>("DEFAULT_CONTROL_TO_FIRE_LATENCY"));
    return base::get_param<double>("auto-aim.latency.control-to-fire");
}

double CoordConverter::get_fire_to_hit_latency(const Eigen::Vector3d& aim_xyz_i_barrel) const {
    double bs = this->get_bullet_speed();
    // 不考虑空气阻力
    return aim_xyz_i_barrel.norm() / bs;
}

double CoordConverter::get_img_to_prediction_latency(const Eigen::Vector3d& aim_xyz_i_barrel
) const {
    return this->get_img_to_predict_latency() + this->get_predict_to_send_latency()
        + this->get_send_to_control_latency() + this->get_fire_to_hit_latency(aim_xyz_i_barrel);
}

// yaw_v 和 原始 yaw 均采用该时间点的信息
double CoordConverter::get_prediction_time(const Eigen::Vector3d& aim_xyz_i_barrel) const {
    return this->get_img_t() + this->get_img_to_prediction_latency(aim_xyz_i_barrel);
}

// 此刻给出发弹指令，目标为 aim_pos，则该指令对应子弹的命中延迟
double CoordConverter::get_img_to_hit_latency(const Eigen::Vector3d& aim_xyz_i_barrel) const {
    return this->get_img_to_predict_latency() + this->get_predict_to_send_latency()
        + this->get_send_to_control_latency() + this->get_control_to_fire_latency()
        + this->get_fire_to_hit_latency(aim_xyz_i_barrel);
}

// 此刻给出发弹指令，目标为 aim_pos，则该指令对应子弹的命中时间
double CoordConverter::get_hit_time(const Eigen::Vector3d& aim_xyz_i_barrel) const {
    return this->get_img_t() + this->get_img_to_hit_latency(aim_xyz_i_barrel);
}

double CoordConverter::get_img_to_control_latency() const {
    return this->get_img_to_predict_latency() + this->get_predict_to_send_latency()
        + this->get_send_to_control_latency();
}

double CoordConverter::get_img_to_fire_latency() const {
    return this->get_img_to_predict_latency() + this->get_predict_to_send_latency()
        + this->get_send_to_control_latency() + this->get_control_to_fire_latency();
}

Eigen::Vector3d CoordConverter::get_camera_z_i() const {
    // store? call? read? write? compile?
    return this->pc_to_pi(Eigen::Vector3d(0., 0., 1.)).normalized();
}

Eigen::Vector2d CoordConverter::get_camera_z_i_pitch_vec() const {
    Eigen::Vector3d camera_z_i = this->get_camera_z_i();
    Eigen::Vector2d camera_z_i_pitch_vec {
        std::sqrt(aimer::math::sq(camera_z_i(0, 0)) + aimer::math::sq(camera_z_i(1, 0))),
        camera_z_i(2, 0)
    };
    camera_z_i_pitch_vec.normalize();
    return camera_z_i_pitch_vec;
}

double CoordConverter::get_camera_z_i_pitch() const {
    return aimer::math::get_theta(this->get_camera_z_i_pitch_vec());
}

// 相机 z 轴在在大地的投影的 normalized
// 又称 get_camera_z_w_yaw_vec
Eigen::Vector2d CoordConverter::get_camera_z_i2() const {
    Eigen::Vector3d camera_z_i = this->get_camera_z_i();
    Eigen::Vector2d camera_z_i2 { camera_z_i(0, 0), camera_z_i(1, 0) };
    camera_z_i2.normalize();
    return camera_z_i2;
}

// 相机 z 轴在大地的投影的极角
double CoordConverter::get_camera_z_i_yaw() const {
    return aimer::math::get_theta(this->get_camera_z_i2());
}

void CoordConverter::catch_predict_timestamp() {
    this->predict_timestamp_binder.update(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        )
                .count()
            / 1e3,
        this->get_frame()
    ); // by agx
}

void CoordConverter::catch_send_timestamp() { // and latency
    double send_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()
                            )
                                .count()
        / 1e3; // 目前与陀螺仪的触发频率统一
    this->predict_to_send_latency_filter.update(
        send_timestamp - this->predict_timestamp_binder.get(this->get_frame()),
        this->get_img_t(),
        { 1. },
        { 3000. }
    );
} // 统计估计

void CoordConverter::write_log(const char* str) const {
    FILE* file = std::fopen(this->file_str.c_str(), "a+");
    std::fprintf(file, "%s\n", str);
    std::fclose(file);
}

double CoordConverter::aim_to_horiz_diff(const aimer::math::YpdCoord& ypd) const {
    // ypd.yaw 是相机坐标系，向右为正
    // yaw_compensate 是电控坐标系，向左为正
    // 操作手让电控自瞄向左转 3 度
    // 电控在视觉传入的数值上 +3 度
    // 视觉发送为 -3 度时，不动
    // 此时视觉需要认为 diff = 0.0
    // cmd.yaw = -3 => aim.yaw = 3
    // aim.yaw - yaw_compensate = 0.0
    double yaw_diff = std::fabs(ypd.yaw - this->get_yaw_compensate());
    return aimer::math::get_termination_dis(ypd.dis, yaw_diff);
}

double CoordConverter::aim_to_vert_diff(const aimer::math::YpdCoord& ypd) const {
    double pitch_diff = std::fabs(ypd.pitch - this->get_pitch_compensate());
    return aimer::math::get_termination_dis(ypd.dis, pitch_diff);
}

// 监督 yaw pitch 类似于 TLE 简称 AEE
bool CoordConverter::aim_error_exceeded(
    const aimer::math::YpdCoord& ypd,
    const aimer::SampleArmor& sample_armor,
    const double& error_rate,
    const double& yaw_inclined_camera,
    // 通常我们方便测的相对 yaw 和绝对 pitch
    const double& pitch_inclined_imu
) const {
    // 当 inclined > 90. 时，error 为负数，必超过 error
    // 其实看到的 pitch_inclined 不一定就等于他对大地的 inclined
    // 事关 robot 的仰角问题
    // 当 error_rate = 1.0 时，极限为装甲板的边缘
    if (this->aim_to_horiz_diff(ypd)
        > sample_armor.width / 2. * std::cos(yaw_inclined_camera) * error_rate)
    {
        return true;
    }
    if (this->aim_to_vert_diff(ypd) > sample_armor.height / 2.
            * std::cos(pitch_inclined_imu + this->get_camera_z_i_pitch()) * error_rate)
    {
        // 对方朝上 15 度，我朝下 15 度，这是刚好对准的
        return true;
    }
    return false;
}

bool CoordConverter::aim_error_exceeded(
    const aimer::math::YpdCoord& real_ypd,
    const aimer::math::YpdCoord& ideal_ypd,
    const aimer::SampleArmor& sample_armor,
    const double& error_rate,
    const double& yaw_inclined_camera,
    // 通常我们方便测的相对 yaw 和绝对 pitch
    const double& pitch_inclined_world
) const {
    if (aimer::math::get_termination_dis(ideal_ypd.dis, real_ypd.yaw, ideal_ypd.yaw)
        > sample_armor.width / 2. * std::cos(yaw_inclined_camera) * error_rate)
    {
        return true;
    }
    if (aimer::math::get_termination_dis(ideal_ypd.dis, real_ypd.pitch, ideal_ypd.pitch)
        > sample_armor.height / 2. * std::cos(pitch_inclined_world + this->get_camera_z_i_pitch())
            * error_rate)
    {
        // 对方朝上 15 度，我朝下 15 度，这是刚好对准的
        return true;
    }
    return false;
}

// 对于 send.yaw, 当 send.yaw + yaw_compensate == 0. 时最佳
// aim.yaw 和 send.yaw 的方向相反
double CoordConverter::aim_swing_cost(const aimer::math::YpdCoord& ypd) const {
    return aimer::math::get_norm(
        ypd.yaw - this->get_yaw_compensate(), // aim = -send
        ypd.pitch - this->get_pitch_compensate()
    );
    // compensate 调大后，我发向下 5 度（+5）时它不动，则枪口是抬高的
}

bool CoordConverter::aim_cmp(const aimer::AimInfo& aim_a, const aimer::AimInfo& aim_b) const {
    if ((aim_a.shoot == ::ShootMode::IDLE) != (aim_b.shoot == ::ShootMode::IDLE)) {
        return static_cast<int>(aim_a.shoot == ::ShootMode::IDLE)
            < static_cast<int>(aim_b.shoot == ::ShootMode::IDLE); // false 优先
    }
    return this->aim_swing_cost(aim_a.ypd) < this->aim_swing_cost(aim_b.ypd);
}

// 通过相机的陀螺仪姿态，求出陀螺仪坐标系
// 前提：传入的 pc 所对应的姿态是真的相机姿态
// 相机 xyz -> 陀螺仪 xyz
Eigen::Vector3d CoordConverter::pc_to_pi(const Eigen::Vector3d& pc) const {
    auto rot_ci = (this->rot_ic_sup * this->rot_ic_q).transpose();
    // R_IW 来自 q，表示旋转
    // R_CI 是标定得到的 3 * 3 矩阵
    return rot_ci * pc;
}

// 陀螺仪 xyz -> 相机 xyz，z朝前
Eigen::Vector3d CoordConverter::pi_to_pc(const Eigen::Vector3d& pi) const {
    auto rot_ic = this->rot_ic_sup * this->rot_ic_q;
    // R_IW 居然是 WORLD_TO_IMU
    return rot_ic * pi;
}

// pu 是 camera_pu, pc 是枪口 pc，pi 也是枪口 pi
// pu 有问题，pu 直接求 pc 是相机 pc
// 相机（实际上是枪口） xyz -> 图像 xy（中心点）
// 不矫正畸变
cv::Point2f CoordConverter::pc_to_pu(const Eigen::Vector3d& pc) const {
    Eigen::Vector3d pu_eigen = this->f_mat * pc / pc(2, 0);
    return cv::Point2f { (float)pu_eigen(0, 0), (float)pu_eigen(1, 0) };
}

Eigen::Vector3d CoordConverter::pu_to_pc_norm(const cv::Point2f& pu) const {
    Eigen::Vector3d pu_vec { pu.x, pu.y, 1. };
    Eigen::Vector3d pc_norm = this->f_mat.inverse() * pu_vec;
    return pc_norm;
}

aimer::math::YpdCoord CoordConverter::pu_to_yp_c(const cv::Point2f& pu) const {
    return aimer::math::camera_xyz_to_ypd(this->pu_to_pc_norm(pu));
}

cv::Point2f CoordConverter::pu_to_pd(const cv::Point2f& pu) const {
    std::vector<cv::Point2f> pus = { pu };
    std::vector<cv::Point2f> pds = {};
    aimer::math::distort_points(pus, pds, this->f_cv_mat, this->c_cv_mat);
    return pds[0];
}

cv::Point2f CoordConverter::pd_to_pu(const cv::Point2f& pd) const {
    std::vector<cv::Point2f> pds = { pd };
    std::vector<cv::Point2f> pus = {};
    cv::undistortPoints(pds, pus, this->f_cv_mat, this->c_cv_mat, cv::noArray(), this->f_cv_mat);
    return pus[0];
}

cv::Point2f CoordConverter::pi_to_pu(const Eigen::Vector3d& pi) const {
    return this->pc_to_pu(this->pi_to_pc(pi));
}

cv::Point2f CoordConverter::pi_to_pd(const Eigen::Vector3d& pi) const {
    return this->pu_to_pd(this->pc_to_pu(this->pi_to_pc(pi)));
}

aimer::math::YpdCoord CoordConverter::pd_to_yp_c(const cv::Point2f& pd) const {
    return this->pu_to_yp_c(this->pd_to_pu(pd));
}

// 此为利用相机的坐标系系统所求出的近似值
aimer::math::YpdCoord CoordConverter::get_camera_ypd_v(
    const Eigen::Vector3d& xyz_i,
    const Eigen::Vector3d& xyz_v_i
) const {
    return aimer::math::camera_get_ypd_v(this->pi_to_pc(xyz_i), this->pi_to_pc(xyz_v_i));
}

// pts 转标准化 pis
std::vector<Eigen::Vector3d> CoordConverter::pts_to_pis_norm(const std::vector<cv::Point2f>& pts
) const {
    std::vector<cv::Point2f> pus;
    cv::undistortPoints(pts, pus, this->f_cv_mat, this->c_cv_mat, cv::noArray(), this->f_cv_mat);
    std::vector<Eigen::Vector3d> pis_norm;
    for (auto& pu: pus) {
        pis_norm.push_back(this->pc_to_pi(this->pu_to_pc_norm(pu)));
    }
    return pis_norm;
}

cv::Point2f CoordConverter::aim_ypd_to_pu(const aimer::math::YpdCoord& ypd) const {
    Eigen::Vector3d pc = aimer::math::camera_ypd_to_xyz(ypd);
    return this->pc_to_pu(pc);
}

Eigen::Vector3d CoordConverter::offset_parabola(const Eigen::Vector3d& pos) const {
    Eigen::Vector3d res = pos;
    aimer::math::YpdCoord ypd = aimer::math::xyz_to_ypd(pos);
    double bs = this->get_bullet_speed();
    double dis = pos.norm(); // = ypd.dis
    double a = 9.8 * 9.8 * 0.25;
    double b = -bs * bs - dis * 9.8 * std::cos(M_PI_2 + ypd.pitch);
    double c = dis * dis;
    double t_2 = (-std::sqrt(b * b - 4. * a * c) - b) / (2. * a);
    // double fly_time = std::sqrt(t_2);
    double height = 0.5 * 9.8 * t_2;
    res(2, 0) += height;
    return res;
}

// 对相机 ypd 进行抬头补偿，给 Enemy 的决策机用即可
// AimParam 可供复现
// aimer::ShootParam CoordConverter::offset_fall(
//     const Eigen::Vector3d& raw_pos) const {
//   // 暂时不放出这个函数，以免滥用
// }

// // 注意修正的是物体坐标
// aimer::math::YpdCoord CoordConverter::offset_camera_to_barrel(
//     const aimer::math::YpdCoord& ypd_camera) const {
//   // 修正的是物体坐标，坐标系向左转时，物体坐标向右转
//   aimer::math::YpdCoord ypd_fix_yaw_pitch(
//       ypd_camera.yaw -
//           base::get_param<double>("launching-mechanism.camera-to-barrel-yAW") / 180. *
//           M_PI,
//       ypd_camera.pitch -
//           base::get_param<double>("CAMERA_TO_BARREL_PITCH") / 180. *
//           M_PI,
//       ypd_camera.dis);
//   Eigen::Vector3d pc_fix_yaw_pitch =
//   aimer::math::camera_ypd_to_xyz(ypd_fix_yaw_pitch); Eigen::Vector3d
//   pc_barrel = pc_fix_yaw_pitch; pc_barrel(1, 0) -=
//   base::get_param<double>("launching-mechanism.camera-to-barrel-y");
//   aimer::math::YpdCoord ypd_barrel =
//   aimer::math::camera_xyz_to_ypd(pc_barrel); return ypd_barrel;
//   // 相机到枪管的 y 加若干，则对于枪管，目标物的坐标 y 减去若干
// }

// aimer::math::YpdCoord CoordConverter::offset_barrel_to_camera(
//     const aimer::math::YpdCoord& ypd_barrel) const {
//   Eigen::Vector3d pc_barrel = aimer::math::camera_ypd_to_xyz(ypd_barrel);
//   Eigen::Vector3d pc_fix_y = pc_barrel;
//   pc_fix_y(1, 0) += base::get_param<double>("launching-mechanism.camera-to-barrel-y");
//   aimer::math::YpdCoord ypd_fix_y = aimer::math::camera_xyz_to_ypd(pc_fix_y);
//   aimer::math::YpdCoord ypd_camera(
//       ypd_fix_y.yaw +
//           base::get_param<double>("launching-mechanism.camera-to-barrel-yAW") / 180. *
//           M_PI,
//       ypd_fix_y.pitch +
//           base::get_param<double>("CAMERA_TO_BARREL_PITCH") / 180. *
//           M_PI,
//       ypd_fix_y.dis);
//   return ypd_camera;
// }

Eigen::Vector3d CoordConverter::xyz_i_camera_to_xyz_i_barrel(const Eigen::Vector3d& xyz_i_camera
) const {
    Eigen::Vector3d xyz_c_camera = this->pi_to_pc(xyz_i_camera);
    // 注意是目标坐标，平移方向与相机平移到枪口相反
    Eigen::Vector3d xyz_c_barrel = {
        xyz_c_camera(0, 0) - base::get_param<double>("launching-mechanism.camera-to-barrel-x"),
        xyz_c_camera(1, 0) - base::get_param<double>("launching-mechanism.camera-to-barrel-y"),
        xyz_c_camera(2, 0)
    };
    // 以枪口为原点的世界坐标系
    Eigen::Vector3d xyz_i_barrel = this->pc_to_pi(xyz_c_barrel);
    return xyz_i_barrel;
}

Eigen::Vector3d CoordConverter::xyz_i_barrel_to_xyz_i_camera(const Eigen::Vector3d& xyz_i_barrel
) const {
    const Eigen::Vector3d xyz_c_barrel = this->pi_to_pc(xyz_i_barrel);
    const Eigen::Vector3d xyz_c_camera = { Eigen::Vector3d(
        xyz_c_barrel(0, 0) + base::get_param<double>("launching-mechanism.camera-to-barrel-x"),
        xyz_c_barrel(1, 0) + base::get_param<double>("launching-mechanism.camera-to-barrel-y"),
        xyz_c_barrel(2, 0)
    ) };
    const Eigen::Vector3d xyz_i_camera = { this->pc_to_pi(xyz_c_camera) };
    return xyz_i_camera;
}

aimer::ShootParam CoordConverter::target_pos_to_shoot_param(const Eigen::Vector3d& target_pos
) const {
    // Eigen::Vector3d pos = raw_pos;
    Eigen::Vector3d target_xyz_i_barrel = this->xyz_i_camera_to_xyz_i_barrel(target_pos);
    const double bs = this->get_bullet_speed();
    const double target_xy = std::sqrt(
        target_xyz_i_barrel(0, 0) * target_xyz_i_barrel(0, 0)
        + target_xyz_i_barrel(1, 0) * target_xyz_i_barrel(1, 0)
    );
    const double target_z = target_xyz_i_barrel(2, 0);
    const double shoot_angle = [&target_xy, &target_z, &bs]() {
        double shoot_angle = 30. / 180. * M_PI;
        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ResistanceFuncLinear, 1, 1>(
                new ResistanceFuncLinear(target_xy, target_z, bs)
            ),
            nullptr,
            &shoot_angle
        );
        ceres::Solver::Options options;
        options.max_num_iterations = 25;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        return shoot_angle;
    }();
    Eigen::Vector3d aim_xyz_i_barrel = target_xyz_i_barrel;
    aim_xyz_i_barrel(2, 0) = target_xy * std::tan(shoot_angle);
    return aimer::ShootParam { bs, shoot_angle, aim_xyz_i_barrel, target_pos };
    // return this->offset_fall();
}

aimer::math::YpdCoord CoordConverter::aim_xyz_i_to_aim_ypd(const Eigen::Vector3d& aim_xyz_i) const {
    Eigen::Vector3d aim_xyz_c_barrel = this->pi_to_pc(aim_xyz_i);
    aimer::math::YpdCoord aim_ypd_c_barrel = aimer::math::camera_xyz_to_ypd(aim_xyz_c_barrel);
    aimer::math::YpdCoord aim_ypd_b {
        aim_ypd_c_barrel.yaw - base::get_param<double>("auto-aim.aim-offset.yaw") / 180. * M_PI,
        aim_ypd_c_barrel.pitch - base::get_param<double>("auto-aim.aim-offset.pitch") / 180. * M_PI,
        aim_ypd_c_barrel.dis
    };
    return aim_ypd_b;
}

// 5 个坐标系：
// 相机坐标系 _c
// 以相机光心为原点的陀螺仪坐标系 _i
// 以枪口中心为原点的陀螺仪坐标系 _i_barrel
// 以枪口中心为原点、相机陀螺仪为方向的坐标系 _c_barrel
// 枪口坐标系（发射坐标系）_b
// .
// 读入后，所有均按照相机坐标系下滤波
// 需在以枪口中心为原点的世界坐标系下做 offset_fall
// 注意此时并不需要补偿相机到枪口的 yaw pitch
// target_pos 是相对于相机的，aim_pos, aim_ypd 是相对于枪口的
aimer::math::YpdCoord CoordConverter::target_pos_to_aim_ypd(const Eigen::Vector3d& pos) const {
    Eigen::Vector3d aim_xyz_i_barrel = this->target_pos_to_shoot_param(pos).aim_xyz_i_barrel;
    return this->aim_xyz_i_to_aim_ypd(aim_xyz_i_barrel);
    // return this->offset_camera_to_barrel(aimer::math::camera_xyz_to_ypd(pc));
}

double CoordConverter::filter_to_prediction_time(const PositionPredictorInterface& filter) const {
    int iterations_num = base::get_param<int64_t>("auto-aim.predict.num-iterations");
    double prediction_t = this->get_img_t();
    for (int i = 0; i < iterations_num; ++i) {
        prediction_t = this->get_prediction_time(
            this->target_pos_to_shoot_param(filter.predict_pos(prediction_t)).aim_xyz_i_barrel
        );
    }
    return prediction_t;
}

aimer::ShootParam CoordConverter::filter_to_shoot_param(const PositionPredictorInterface& filter
) const {
    return this->target_pos_to_shoot_param(filter.predict_pos(this->filter_to_prediction_time(filter
    )));
}

// aimer::math::YpdCoord CoordConverter::filter_to_aim_ypd(
//     const PositionPredictorInterface& filter) const {
//   return this->target_pos_to_aim_ypd(this->filter_to_shoot_param(filter));
// }

// aim 总是关于枪口的，这里存在误差
aimer::math::YpdCoord CoordConverter::filter_to_aim_ypd_v(const PositionPredictorInterface& filter
) const {
    double prediction_t = this->filter_to_prediction_time(filter);
    return this->get_camera_ypd_v(filter.predict_pos(prediction_t), filter.predict_v(prediction_t));
}

double CoordConverter::filter_to_hit_time(const PositionPredictorInterface& filter) const {
    int iterations_num = base::get_param<int64_t>("auto-aim.predict.num-iterations");
    double hit_t = this->get_img_t();
    for (int i = 0; i < iterations_num; ++i) {
        hit_t = this->get_hit_time(
            this->target_pos_to_shoot_param(filter.predict_pos(hit_t)).aim_xyz_i_barrel
        );
    }
    return hit_t;
}

Eigen::Vector3d CoordConverter::filter_to_hit_pos(const PositionPredictorInterface& filter) const {
    return filter.predict_pos(this->filter_to_hit_time(filter));
}

aimer::math::YpdCoord CoordConverter::filter_to_hit_aim_ypd(const PositionPredictorInterface& filter
) const {
    return this->target_pos_to_aim_ypd(this->filter_to_hit_pos(filter));
}

// aim 总是关于枪口的，这里存在误差
aimer::math::YpdCoord
CoordConverter::filter_to_hit_aim_ypd_v(const PositionPredictorInterface& filter) const {
    double prediction_t = this->filter_to_hit_time(filter);
    return this->get_camera_ypd_v(filter.predict_pos(prediction_t), filter.predict_v(prediction_t));
}

double CoordConverter::get_control_aim_yaw0() const {
    return this->get_yaw_compensate();
}

double CoordConverter::get_control_aim_pitch0() const {
    return this->get_pitch_compensate();
}

void CoordConverter::update(
    const cv::Mat& img,
    const Eigen::Quaternionf& q,
    const double& timestamp
) {
    this->frame += 1;
    this->img = img;
    this->q = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());
    this->rot_ic_q = q.matrix().transpose().cast<double>();
    this->img_t = timestamp; // img_time_camera

    // 根据四元数更新本帧图像所对应的  动态旋转矩阵
    // R_IW 在函数里是 世界 - 陀螺仪（注意这边的陀螺仪是相机坐标系方向）
    // 相机和陀螺仪仅有安装偏差
    // 每次预测采样一次 cmd_to_fire latency
    // this->latency_cmd_to_fire_filter.update(
    //     this->get_robot_status().latency_cmd_to_fire / 1e3,
    //     this->img_t);
}

/** @class CreditClock */

CreditClock::CreditClock(aimer::CoordConverter* const converter, const double& credit_time):
    converter(converter),
    credit_time(credit_time) {}

void CreditClock::update() {
    this->update_t = this->converter->get_img_t();
}

bool CreditClock::credit() const {
    return this->converter->get_img_t() - this->update_t <= this->credit_time;
}

double CreditClock::get_update_t() const {
    return this->update_t;
}

} // namespace aimer
