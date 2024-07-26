#include "aimer/auto_aim/predictor/enemy/outpost_fixer.hpp"

#include "base/param/parameter.hpp"

namespace aimer {

std::vector<ArmorData>
OutpostFixer::filter(const std::vector<ArmorData>& armors, const double& request_t) {
    const double z_that_distinguishes =
        base::get_param<double>("auto-aim.outpost-fixer.z-that-distinguishes-two-types");
    const double active_lasting_time =
        base::get_param<double>("auto-aim.outpost-fixer.active-lasting-time");
    auto get_low_z = [&](const std::vector<ArmorData>& armors) -> double {
        double low_z = aimer::INF;
        for (const auto& armor: armors) {
            low_z = std::min(low_z, armor.info.pos[2]);
        }
        return low_z;
    };

    // 场景 1：低处（无干扰）
    // 场景 2：低处走到高处，高处走到低处
    // 场景 3：低处不看，走到高处，高处不看，走到低处
    // 场景 4：高处，第一帧只有顶部，腰部定时转过去看不见，顶部一直闪烁

    // 不活跃，直接取最低，变活跃，然后筛
    // 活跃，则利用上一活跃的最低，筛高的不合法
    // 若本帧有合法，更新活跃
    // 持续看时，最低不应成为永久限制

    if (this->last_active_time + active_lasting_time < request_t) {
        if (!armors.empty()) {
            this->last_active_z = get_low_z(armors);
            this->last_active_time = request_t;
        }
    }

    auto filtered = armors;
    for (auto it = filtered.begin(); it != filtered.end();) {
        if (this->last_active_z + z_that_distinguishes < it->info.pos[2]) {
            it = filtered.erase(it);
        } else {
            it++;
        }
    }

    if (!filtered.empty()) {
        this->last_active_z = get_low_z(filtered);
        this->last_active_time = request_t;
    }
    return filtered;
}

} // namespace aimer
