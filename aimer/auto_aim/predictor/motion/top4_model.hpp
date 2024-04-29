#ifndef AIMER_AUTO_AIM_PREDICTOR_MOTION_TOP4_MODEL_HPP
#define AIMER_AUTO_AIM_PREDICTOR_MOTION_TOP4_MODEL_HPP

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/predictor/motion/top_model.hpp"

namespace aimer::top::top4 {

// 描述在视野中的两块装甲板
enum class TopPattern { AB, B, BA, A };

// 描述两种不同板的信息
struct AbArmor {
    AbArmor(const double& length, const double& z_plus) {
        this->length = length;
        this->z_plus = z_plus;
    }
    double length{0.};
    // 相对于车中心的高度
    double z_plus{0.};
};

// 记录两种不同板的信息
class AbManager {
  public:
    top4::AbArmor get_ab() const;

    void update(const Eigen::Vector3d& center, const top4::AbArmor& d);

    double get_cost(const top4::AbArmor& d) const;

  private:
    SingleFilter<1> length_filter;
    SingleFilter<1> z_filter;
};

// 装甲板变化方向判断机，并不会用于旋转方向判断
class IdDirectionJudger {
  public:
    top::TopDirection get() const;
    void update(const std::vector<aimer::ArmorData>& armors);

  private:
    top::TopDirection direction = top::TopDirection::CCW;
    std::vector<aimer::ArmorData> last_armors;
};

// 陀螺角度预测机
class RotateCounter {
  public:
    explicit RotateCounter(const double& init_radius);

    top4::TopPattern get_pattern() const;
    int get_active_rotate() const;
    // 传入 0 返回 A 板，传入 1 返回 B 板
    AbArmor get_ab(const int& index) const;
    double get_w() const;
    // 外部传进的数据宽容对待，我给外部的数据严格优秀
    double predict_theta_l(const double& t) const;
    bool w_active() const;
    bool w_inactive() const;
    bool top_active() const;

    void update_theta_l(const double& theta_l, const double& t, const double& R);
    void update_rotate(const std::vector<aimer::ArmorData>& armors);
    // 根据长度和高低对 AB 进行检查和修正，仅在部分采样点这样做
    void update_ab(const Eigen::Vector3d& center, const std::vector<AbArmor>& ab, const double& t);

  private:
    TopPattern pattern{TopPattern::AB};
    IdDirectionJudger id_direction_judger;
    Kalman<1, 2> pattern_fixer{{0.01, 10.}, 25, 2};  // 此处参数设置需十分小心
    // [0] 存 A, [1] 存 B
    AbManager ab_manager[2];
    AngleFilter<1, 2> theta_s_filter{2. * M_PI, {0.01, 10., 10.}, aimer::RESERVED_DOUBLE};
    int rotate{0};  // l 板到 super 板需要旋转多少个 90 度
    int active_rotate{0};  // 活跃旋转次数
};

class TopModel {
  public:
    TopModel(
        aimer::CoordConverter* const converter,
        aimer::EnemyState* const state,
        const double& credit_time);
    // 少了少了，常数又少啦
    ~TopModel() = default;

    void update();

    bool active() const;
    std::vector<TopArmor> predict_armors(const double&) const;

    aimer::AimInfo get_limit_aim() const;
    void draw_aim(cv::Mat&) const;

  private:
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    aimer::CreditClock credit_clock;
    top::CenterFilter center_filter;
    top4::RotateCounter counter;

    // 签名是私有的
    struct LengthSample {
        LengthSample(
            const Eigen::Vector3d& center,
            const double& z_to_l_fit,
            const std::vector<top4::AbArmor>& ab) :
            center(center),
            z_to_l_fit(z_to_l_fit),
            ab{ab} {}
        Eigen::Vector3d center;
        double z_to_l_fit;
        std::vector<top4::AbArmor> ab;
    };

    void update_by_double_armors();
    LengthSample length_sampling(
        const std::vector<aimer::ArmorData>&,
        const double&,
        aimer::CoordConverter*) const;
    double double_angle_sampling(
        const std::vector<aimer::ArmorData>&,
        const double&,
        aimer::CoordConverter*) const;

    void update_by_single_armor();
};

}  // namespace aimer::top::top4

#endif /* AIMER_AUTO_AIM_PREDICTOR_MOTION_TOP4_MODEL_HPP */
