/*存储外部信息（含系统内对系统外的修正）*/
#ifndef AIMER_BASE_ROBOT_COORD_CONVERTER_HPP
#define AIMER_BASE_ROBOT_COORD_CONVERTER_HPP

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "UltraMultiThread/include/umt/umt.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/debug/debug.hpp"
#include "aimer/base/math/filter/filter.hpp"
#include "aimer/base/math/math.hpp"
#include "base/param/parameter.hpp"
// robot.hpp 应被移除
#include "core_io/robot.hpp"

namespace aimer {
// 系统信息

// 用于复现的瞄准参数
struct ShootParam {
    double v0 = 0.;
    double aim_angle = 0.;
    Eigen::Vector3d aim_xyz_i_barrel = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_xyz_i_camera = Eigen::Vector3d::Zero();
    // Eigen::Vector3d target_pos = Eigen::Vector3d();
};

struct AimInfo {
    aimer::math::YpdCoord ypd;
    aimer::math::YpdCoord ypd_v;
    // 若此命令发射，且 ypd = 0，则落点为
    // 若此命令发射，则发射参数为
    aimer::ShootParam shoot_param;
    ::ShootMode shoot = ::ShootMode::IDLE;
    int info = 0;

    AimInfo(
        const aimer::math::YpdCoord& ypd,
        const aimer::math::YpdCoord& ypd_v,
        const aimer::ShootParam& shoot_param,
        const ::ShootMode& shoot
    ):
        ypd { ypd },
        ypd_v { ypd_v },
        shoot_param { shoot_param },
        shoot { shoot } {}
    enum : int { TOP = 1 << 0 };
    static const AimInfo idle() {
        return AimInfo(
            aimer::math::YpdCoord(),
            aimer::math::YpdCoord(),
            aimer::ShootParam(),
            ::ShootMode::IDLE
        );
    }
    // convention: 0 位 (1 << 0)，1 表示在陀螺
};

class PnpDistanceFixer {
public:
    PnpDistanceFixer(const double& a0, const double& a1, const double& a2):
        a0(a0),
        a1(a1),
        a2(a2) {}
    [[deprecated]] double fixed_dis(const double& dis) const {
        return this->a0 + this->a1 * dis + this->a2 * dis * dis;
    }

private:
    const double a0;
    const double a1;
    const double a2;
};

template<typename T>
class FrameBinder {
public:
    void update(const T& value, const int& frame) {
        this->value = value;
        this->frame = frame;
    }
    T get(const int& frame) const {
        assert(this->frame == frame);
        return value;
    }

private:
    T value = T();
    int frame = 0;
};

// 注意 pts 和 pus
// pc， pi 始终在相机系统下
// 仅对于 aim_ypd 改用枪管系统
// 但注意，img 和四元数唯一对应，看到反向拟合的应该是对的

class CoordConverter {
private:
    // 相机参数和陀螺仪状态
    Eigen::Matrix3d rot_ic_sup; // 陀螺仪坐标系到相机坐标系旋转矩阵EIGEN-Matrix
    Eigen::Matrix3d f_mat; // 相机内参矩阵EIGEN-Matrix
    Eigen::Matrix<double, 1, 5> c_mat; // 相机畸变矩阵EIGEN-Matrix
    cv::Mat rot_ic_sup_cv_mat; // 陀螺仪坐标系到相机坐标系旋转矩阵CV-Mat
    cv::Mat f_cv_mat; // 相机内参矩阵CV-Mat
    cv::Mat c_cv_mat; // 相机畸变矩阵CV-Mat
    Eigen::Quaterniond q;
    Eigen::Matrix3d rot_ic_q; // 相机 - 世界坐标系旋转所需矩阵

    std::shared_ptr<::RobotStatus>
        robot_status_obj; // 由 /core_io 中 Serial port （串行接口） 读入并
    // memcpy 进去的公共指针。创建 converter 时即绑定
    // 该存储结构的行为：
    // 最多存储 100 个数据
    // 仅供外部查询四元数使用

    cv::Mat img;
    int frame = 0; // 帧计数
    double img_t = 0.; // 相机图像时间

    std::string file_str;

    // from detection_result 我得到 img_timestamp
    // double predict_timestamp = 0.;  // s
    aimer::FrameBinder<double> predict_timestamp_binder;
    // double send_timestamp = 0.;     // s estimate
    // always make reader understand!
    aimer::SingleFilter<1> predict_to_send_latency_filter {};

public:
    // 加载相机参数
    CoordConverter(); // extern

    ~CoordConverter() = default;

    const cv::Mat& get_img_ref() const;
    // 获取帧计数
    int get_frame() const;
    // 输出图像对应的相机时间
    double get_img_t() const;

    // 获取电控发送的 "RobotStatus" 机器人状态信息
    const RobotStatus& get_robot_status_ref() const;
    // 获取电控实际使用的 yaw 角补偿
    double get_yaw_compensate() const;
    double get_pitch_compensate() const;
    double get_bullet_speed() const;

    double get_img_to_predict_latency() const;
    double get_predict_to_send_latency() const;
    double get_send_to_control_latency() const;
    double get_control_to_fire_latency() const;
    double get_fire_to_hit_latency(const Eigen::Vector3d& aim_xyz_i_barrel) const;
    // 获取预测时间量，推导见 aimer/docs/latency.md
    double get_img_to_prediction_latency(const Eigen::Vector3d& aim_xyz_i_barrel) const;
    double get_prediction_time(const Eigen::Vector3d& aim_xyz_i_barrel) const;
    double get_img_to_hit_latency(const Eigen::Vector3d& aim_xyz_i_barrel) const;
    double get_hit_time(const Eigen::Vector3d& aim_xyz_i_barrel) const;
    double get_img_to_control_latency() const;
    double get_img_to_fire_latency() const;

    // 获取相机 z 轴单位向量在陀螺仪坐标系下的表示
    Eigen::Vector3d get_camera_z_i() const;
    Eigen::Vector2d get_camera_z_i_pitch_vec() const;
    double get_camera_z_i_pitch() const;
    // 相机 z 轴在在大地的投影的 normalized
    // 又称 get_camera_z_w_yaw_vec
    Eigen::Vector2d get_camera_z_i2() const;
    // 相机 z 轴在大地的投影的极角
    double get_camera_z_i_yaw() const;

    void update(const cv::Mat& img, const Eigen::Quaternionf& q, const double& timestamp);

    void catch_predict_timestamp();
    void catch_send_timestamp();

    void write_log(const char* str) const;

    double aim_to_horiz_diff(const aimer::math::YpdCoord& ypd) const;
    double aim_to_vert_diff(const aimer::math::YpdCoord& ypd) const;
    // 监督 yaw pitch 类似于 TLE 简称 AEE
    bool aim_error_exceeded(
        const aimer::math::YpdCoord& ypd,
        const aimer::SampleArmor& sample_armor,
        const double& error_rate,
        const double& yaw_inclined_camera,
        // 通常我们方便测的相对 yaw 和绝对 pitch
        const double& pitch_inclined_imu
    ) const;
    bool aim_error_exceeded(
        const aimer::math::YpdCoord& aim_ypd,
        const aimer::math::YpdCoord& armor_ypd,
        const aimer::SampleArmor& sample_armor,
        const double& error_rate,
        const double& yaw_inclined_camera,
        // 通常我们方便测的相对 yaw 和绝对 pitch
        const double& pitch_inclined_imu
    ) const;
    // 对于 send.yaw, 当 send.yaw + yaw_compensate == 0. 时最佳
    // aim.yaw 和 send.yaw 的方向相反
    double aim_swing_cost(const aimer::math::YpdCoord& ypd) const;
    bool aim_cmp(const aimer::AimInfo& aim_a, const aimer::AimInfo& aim_b) const;

    // 通过相机的陀螺仪姿态，求出陀螺仪坐标系
    // 前提：传入的 pc 所对应的姿态是真的相机姿态
    // 相机 xyz -> 陀螺仪 xyz
    /** @brief xyz Point in Camera coordinate system to xyz Point in World
   * coordinate system */
    Eigen::Vector3d pc_to_pi(const Eigen::Vector3d& pc) const;
    // 陀螺仪 xyz -> 相机 xyz，z朝前
    Eigen::Vector3d pi_to_pc(const Eigen::Vector3d& pi) const;

    // pu 是 camera_pu, pc 是枪口 pc，pi 也是枪口 pi
    // pu 有问题，pu 直接求 pc 是相机 pc
    // 相机（实际上是枪口） xyz -> 图像 xy（中心点）
    // 不矫正畸变
    /** @brief xyz Point in Camera coordinate_system to Point Undistorted */
    cv::Point2f pc_to_pu(const Eigen::Vector3d& pc) const;
    Eigen::Vector3d pu_to_pc_norm(const cv::Point2f& pu) const;
    aimer::math::YpdCoord pu_to_yp_c(const cv::Point2f& pu) const;

    cv::Point2f pu_to_pd(const cv::Point2f& pu) const;
    cv::Point2f pd_to_pu(const cv::Point2f& pd) const;

    cv::Point2f pi_to_pu(const Eigen::Vector3d& pi) const;
    cv::Point2f pi_to_pd(const Eigen::Vector3d& pi) const;
    /** @brief Point Distorted to Yaw Pitch in Camera coordinate system */
    aimer::math::YpdCoord pd_to_yp_c(const cv::Point2f& pd) const;

    // 此为利用相机的坐标系系统所求出的近似值
    aimer::math::YpdCoord
    get_camera_ypd_v(const Eigen::Vector3d& xyz_i, const Eigen::Vector3d& xyz_v_i) const;
    // pts 转标准化 pis
    std::vector<Eigen::Vector3d> pts_to_pis_norm(const std::vector<cv::Point2f>& pts) const;

    cv::Point2f aim_ypd_to_pu(const aimer::math::YpdCoord& ypd) const;

    const cv::Mat& get_f_cv_mat_ref() const {
        return this->f_cv_mat;
    }
    const cv::Mat& get_c_cv_mat_ref() const {
        return this->c_cv_mat;
    }
    const cv::Mat& get_rot_ic_sup_cv_mat_ref() const {
        return this->rot_ic_sup_cv_mat;
    }
    Eigen::Quaterniond get_q() const {
        return this->q;
    }

    /** @brief 古早的抛物线补偿函数 */
    Eigen::Vector3d offset_parabola(const Eigen::Vector3d& pos) const;
    // 对相机 ypd 进行抬头补偿，给 Enemy 的决策机用即可
    // ShootParam offset_fall(const Eigen::Vector3d& raw_pos) const;
    // 输入 dis，输出视觉发送 shoot
    // 信号到子弹击中装甲板的预计时间（需要子弹初速度系统信息）
    // 注意修正的是物体坐标
    // aimer::math::YpdCoord offset_camera_to_barrel(const aimer::math::YpdCoord&
    // ypd_camera) const; aimer::math::YpdCoord offset_barrel_to_camera(const
    // aimer::math::YpdCoord& ypd_barrel) const;

    /**
   * @brief 把 坐标系 [原点: 相机, 方向: 陀螺仪] 下的坐标转为
   * 坐标系 [原点: 枪口, 方向：陀螺仪] 下的坐标
   */
    Eigen::Vector3d xyz_i_camera_to_xyz_i_barrel(const Eigen::Vector3d& xyz_i_camera) const;
    Eigen::Vector3d xyz_i_barrel_to_xyz_i_camera(const Eigen::Vector3d& xyz_i_barrel) const;

    /// @param pos 目标在 [原点: 相机，轴系: 陀螺仪] 坐标系 下的坐标。
    /// @return 枪口需要指向的点在 [原点: 枪口，轴系: 陀螺仪] 坐标系下的坐标
    aimer::ShootParam target_pos_to_shoot_param(const Eigen::Vector3d& target_pos) const;
    // aim 包括了抬头补偿，返回的是相机坐标系下的 ypd，这里的 yaw 是俯视顺时针为正
    // hit_xyz_w_to_aim_ypd_b
    aimer::math::YpdCoord aim_xyz_i_to_aim_ypd(const Eigen::Vector3d& aim_xyz_i) const;
    aimer::math::YpdCoord target_pos_to_aim_ypd(const Eigen::Vector3d& pos) const;
    double filter_to_prediction_time(const PositionPredictorInterface& filter) const;
    // 这个
    // (现在的时间 + 子弹延迟 - filter 中的时间）aim 包括了抬头补偿
    // target_pos is not aim_pos
    // 如果枪口正在无限正确发弹，则发弹位置
    aimer::ShootParam filter_to_shoot_param(const PositionPredictorInterface& filter) const;
    // aimer::math::YpdCoord filter_to_aim_ypd(const PositionPredictorInterface&
    // filter) const;
    aimer::math::YpdCoord filter_to_aim_ypd_v(const PositionPredictorInterface& filter) const;
    double filter_to_hit_time(const PositionPredictorInterface& filter) const;
    Eigen::Vector3d filter_to_hit_pos(const PositionPredictorInterface& filter) const;
    aimer::math::YpdCoord filter_to_hit_aim_ypd(const PositionPredictorInterface& filter) const;
    aimer::math::YpdCoord filter_to_hit_aim_ypd_v(const PositionPredictorInterface& filter) const;

    // 获取使电控不动的 yaw
    double get_control_aim_yaw0() const;
    double get_control_aim_pitch0() const;
    // Kalman<1, 1> latency_cmd_to_fire_filter = // 未保证取样点不重复
    //     Kalman<1, 1>(std::vector<double>{1.}, 3000.);  // double
};

// 无法更新时，可信任时间的倒数器，实现漏帧保护
// 不可信任也不必立即杀死，因此还有留存之功能
class CreditClock {
public:
    CreditClock(aimer::CoordConverter* const converter, const double& credit_time);
    void update();

    bool credit() const;

    double get_update_t() const;

private:
    aimer::CoordConverter* const converter;
    const double credit_time;
    double update_t = 0.;
};

} // namespace aimer

#endif /* AIMER_BASE_ROBOT_COORD_CONVERTER_HPP */
