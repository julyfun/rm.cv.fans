#include "aimer/auto_aim/predictor/enemy/balance.hpp"

#include "base/param/parameter.hpp"

namespace aimer {
/// @class BalanceJudger

void BalanceJudger::update(
    const std::vector<aimer::ArmorData>& armors,
    const double& t,
    const int& frame) {
    // size: 能看到装甲板的数量
    if (armors.empty()) {
        return;
    }
    int credit_size = [](const std::vector<aimer::ArmorData>& armors) {
        int size = 0;
        for (const auto& d: armors) {
            size += int(d.info.detected.conf >= 0.6 && d.info.detected.conf_class >= 0.8);
        }
        return size;
    }(armors);

    if (frame == this->last_frame + 1 && credit_size == this->last_size) {
        if (credit_size >= 2) {
            this->double_conf += 0.5 * (t - this->last_t)
                / base::get_param<double>(
                                     "auto-aim.balance-judger.non-balance-confidence-ramp-time");
        } else {
            this->double_conf -= 0.5 * (t - this->last_t)
                / base::get_param<double>("auto-aim.balance-judger.balance-confidence-ramp-time");
        }
        // if (double_cons == 0)
        //   double_conf -= 0.00033;
        // else
        //   double_conf += (1. - double_conf) *
        //                  aimer::math::sigmoid(double_cons - 2) * 0.2;
    }
    if (this->double_conf < 0.) {
        this->double_conf = 0.;
    }
    if (this->double_conf > 1.) {
        this->double_conf = 1.;
    }
    this->last_frame = frame;
    this->last_t = t; // oh my gosh
    this->last_size = credit_size;
}
} // namespace aimer
