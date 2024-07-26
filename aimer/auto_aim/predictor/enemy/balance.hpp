/*平衡步兵判断模块*/

#ifndef AIMER_AUTO_AIM_PREDICTOR_ENEMY_BALANCE_HPP
#define AIMER_AUTO_AIM_PREDICTOR_ENEMY_BALANCE_HPP

#include <iostream>
#include <vector>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"

namespace aimer {
// 通过可见装甲板信息判断车辆是不是平衡步兵
class BalanceJudger {
  public:
    BalanceJudger() = default;

    void update(const std::vector<aimer::ArmorData>& armors, const double& t, const int& frame);

    bool is_balance() const {
        return this->double_conf > 0.5 ? false : true;
    }

    double get_double_conf() const {
        return this->double_conf;
    }

  private:
    double double_conf = 0.;
    int last_frame = 0;
    double last_t = 0.;  // won't be used at first
    int last_size = 0;
};
}  // namespace aimer

#endif /* AIMER_AUTO_AIM_PREDICTOR_ENEMY_BALANCE_HPP */
