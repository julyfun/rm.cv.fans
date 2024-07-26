#include "aimer/auto_aim/predictor/pnp/pnp.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/base/robot/coord_converter.hpp"
#include "base/param/parameter.hpp"

namespace aimer {

// 四点坐标 + 装甲板信息 -> 相机 ypd，会根据 armor 的数字区分大板和小板
void pnp_get_pc(
    const aimer::DetectedArmor& detected,
    const aimer::SampleArmor& sample,
    const aimer::CoordConverter* const converter,
    Eigen::Vector3d& pc,
    Eigen::Matrix3d& r_mat
) {
    std::vector<cv::Point2d> pts { detected.pts[0],
                                   detected.pts[1],
                                   detected.pts[2],
                                   detected.pts[3] };
    cv::Mat t_vec, r_vec;
    if (sample.type == aimer::ArmorType::BIG) {
        cv::solvePnP(
            aimer::PW_BIG,
            pts,
            converter->get_f_cv_mat_ref(),
            converter->get_c_cv_mat_ref(),
            r_vec,
            t_vec,
            false,
            cv::SOLVEPNP_IPPE
        );
    } else {
        cv::solvePnP(
            aimer::PW_SMALL,
            pts,
            converter->get_f_cv_mat_ref(),
            converter->get_c_cv_mat_ref(),
            r_vec,
            t_vec,
            false,
            cv::SOLVEPNP_IPPE
        );
    }
    cv::Mat cv_r_mat;
    cv::Rodrigues(r_vec, cv_r_mat);
    // 这里还不用修正距离
    cv::cv2eigen(t_vec, pc);
    cv::cv2eigen(cv_r_mat, r_mat);
    return;
}

aimer::ArmorInfo detected_to_info(
    const aimer::DetectedArmor& detected,
    const aimer::SampleArmor& sample,
    const double& armor_pitch,
    const aimer::CoordConverter* const converter
) {
    const std::array<cv::Point2f, 4> pts = { detected.pts[0],
                                             detected.pts[1],
                                             detected.pts[2],
                                             detected.pts[3] };
    const std::array<cv::Point2f, 4> pus;
    cv::undistortPoints(
        pts,
        pus,
        converter->get_f_cv_mat_ref(),
        converter->get_c_cv_mat_ref(),
        cv::noArray(),
        converter->get_f_cv_mat_ref()
    );
    Eigen::Vector3d pc;
    Eigen::Matrix3d r_mat;
    aimer::pnp_get_pc(detected, sample, converter, pc, r_mat);
    const aimer::math::YpdCoord ypd_c = aimer::math::camera_xyz_to_ypd(pc);
    const aimer::PnpDistanceFixer pnp_distance_fixer(
        0.,
        1.,
        base::get_param<double>("auto-aim.pnp.distance-fixer.a2")
    );
    const aimer::math::YpdCoord ypd_c_feature_fixed { ypd_c.yaw,
                                                      ypd_c.pitch,
                                                      pnp_distance_fixer.fixed_dis(ypd_c.dis) };
    const Eigen::Vector3d pc_feature_fixed = aimer::math::camera_ypd_to_xyz(ypd_c_feature_fixed);
    const Eigen::Vector3d pi = converter->pc_to_pi(pc_feature_fixed);
    const aimer::math::YpdCoord yp_i = [&]() {
        // armor-z-axis in the armor coordinate system
        const Eigen::Vector3d armor_z_a = { 0.0, 0.0, 1.0 };
        // r 是相机 - 世界（其实是装甲板）
        const Eigen::Vector3d armor_z_c = r_mat.transpose() * armor_z_a;
        const Eigen::Vector3d armor_z_i = converter->pc_to_pi(armor_z_c);
        return aimer::math::xyz_to_ypd(armor_z_i);
    }();
    return aimer::ArmorInfo { converter->get_frame(),    detected, sample, pus, armor_pitch, pi,
                              Eigen::Quaterniond(r_mat), yp_i };
}

} // namespace aimer
