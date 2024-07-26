#ifndef AIMER_AUTO_AIM_PREDICTOR_ENEMY_ENEMY_STATE_HPP
#define AIMER_AUTO_AIM_PREDICTOR_ENEMY_ENEMY_STATE_HPP

#include <iostream>
#include <map>
#include <vector>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/predictor/enemy/balance.hpp"
#include "aimer/auto_aim/predictor/enemy/outpost_fixer.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"
#include "core_io/robot.hpp"

namespace aimer {
// 当结构过大时，改用文件存储
// 该类并不知道什么是 Model
struct EnemyInfo {
    // 根据常数列表生成的常量表格
    aimer::SampleArmor sample_armor;
    double armor_pitch;
    int armors_num;
    int seen_armors_max_num;
    double default_radius;
    std::string aim_error_param;
}; // number->enemy->sample_armor

// 敌人状态数据库类
class EnemyState {
public:
    EnemyState(const int& number, const aimer::EnemyType& init_type);

    void update(const std::vector<aimer::ArmorData>& raw_armors, const double& t, const int& frame);

    // aimer::EnemyType get_enemy_type() const { return this->enemy_type; }
    aimer::EnemyType get_enemy_type() const;
    const aimer::EnemyInfo& get_info_ref() const;
    int get_armors_num() const;
    // 理论最多可视装甲板
    int get_max_seen() const;
    int get_number() const;
    // 打击宽松程度（返回值为 x 表示将目标装甲板放大 x 倍的范围内允许打击）
    double get_aim_error() const;
    // 获取默认半径
    double get_default_radius() const;
    double get_armor_pitch() const;
    const aimer::SampleArmor& get_sample_armor_ref() const;
    const std::vector<aimer::ArmorData>& get_armor_data_ref() const;

    bool is_hit() const;

private:
    const int number;
    aimer::BalanceJudger balance_judger; // 控制变换
    aimer::EnemyType enemy_type; // 变换敏感
    std::vector<aimer::ArmorData> last_armors;
    std::vector<aimer::ArmorData> armor_data;
    OutpostFixer outpost_fixer;

    void switch_enemy_type(const aimer::EnemyType& enemy_type);

    // 筛除看起来不可能的装甲板
    std::vector<aimer::ArmorData> screened_armors(
        const std::vector<aimer::ArmorData>& raw_armors,
        const std::vector<aimer::ArmorData>& last_armors
    );

    // 根据指定 size 裁剪和排序装甲板
    std::vector<aimer::ArmorData>
    sorted_armors(const std::vector<aimer::ArmorData>& src, const int& max_size);
};
} // namespace aimer

#endif /* AIMER_AUTO_AIM_PREDICTOR_ENEMY_ENEMY_STATE_HPP */
