// [update] 每帧的一辆车 ArmorData Vec。
// Vec 大小必须为 1
// [input] 车 ArmorData Vec
// Vec size 可以任意
// [output] 输出滤出前哨站顶部的装甲板，大小最大为 1，
// 典型的输出为 0，当只有超高装甲板时

// 之前已经有多层过滤
// 放在哪里过滤呢？

// 思路：记录上一帧内合法的装甲板的 z
// 若上次活跃时间为 1s 前，则可以重置了
// 活跃时间更新：当有合法装甲板时
// ...
// 必然使用 filter 的结果更新？

#ifndef AIMER_AUTO_AIM_PREDICTOR_ENEMY_OUTPOST_FIXER_HPP
#define AIMER_AUTO_AIM_PREDICTOR_ENEMY_OUTPOST_FIXER_HPP

#include <vector>

#include "aimer/auto_aim/base/defs.hpp"

namespace aimer {

class OutpostFixer {
public:
    std::vector<ArmorData> filter(const std::vector<ArmorData>& armors, const double& request_t);

private:
    // what state?
    double last_active_time = -10086.0;
    double last_active_z = 1919180.0;
};

} // namespace aimer

#endif
