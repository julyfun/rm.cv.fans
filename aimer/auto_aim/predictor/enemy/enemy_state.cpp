#include "aimer/auto_aim/predictor/enemy/enemy_state.hpp"

#include "base/param/parameter.hpp"

namespace aimer {

const std::string INFANTRY_AIM_MAX_ERROR_STR = "auto-aim.enemy-model.infantry.aim.max-error";
const std::string OLD_SENTRY_AIM_MAX_ERROR_STR = "auto-aim.enemy-model.old-sentry.aim.max-error";

const std::unordered_map<aimer::EnemyType, aimer::EnemyInfo> ENEMY_INFO {
    { aimer::EnemyType::OLD_SENTRY,
      aimer::EnemyInfo { /*sample_armor=*/aimer::BIG_ARMOR,
                         /*armor_pitch=*/aimer::math::deg_to_rad(-15.),
                         /*armors_num=*/2,
                         /*seen_armors_max_num=*/1,
                         /*default_radius=*/0.25,
                         /*aim_error_param=*/OLD_SENTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::HERO,
      aimer::EnemyInfo { aimer::BIG_ARMOR,
                         aimer::math::deg_to_rad(15.),
                         4,
                         2,
                         0.25,
                         INFANTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::ENGINEER,
      aimer::EnemyInfo { aimer::SMALL_ARMOR,
                         aimer::math::deg_to_rad(15.),
                         4,
                         2,
                         0.25,
                         INFANTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::INFANTRY,
      aimer::EnemyInfo { aimer::SMALL_ARMOR,
                         aimer::math::deg_to_rad(15.),
                         4,
                         2,
                         0.25,
                         INFANTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::BALANCE_INFANTRY,
      EnemyInfo { aimer::BIG_ARMOR,
                  aimer::math::deg_to_rad(15.),
                  2,
                  1,
                  0.473 / 2.,
                  INFANTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::OUTPOST,
      EnemyInfo { aimer::SMALL_ARMOR,
                  aimer::math::deg_to_rad(-15.),
                  3,
                  2,
                  0.553 / 2.,
                  INFANTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::CRYSTAL_BIG,
      EnemyInfo { aimer::BIG_ARMOR,
                  aimer::math::deg_to_rad(15.),
                  1,
                  1,
                  0.25,
                  INFANTRY_AIM_MAX_ERROR_STR } },
    { aimer::EnemyType::CRYSTAL_SMALL,
      EnemyInfo { aimer::SMALL_ARMOR,
                  aimer::math::deg_to_rad(15.),
                  1,
                  1,
                  0.25,
                  INFANTRY_AIM_MAX_ERROR_STR } }
};
// 排除极小值（不可信），防止被近处误识别消除，防止抖动排除
const double EXISTING_ARMOR_CRED_AREA_RATIO = 0.30;
const double NEW_ARMOR_CRED_AREA_RATIO = 0.40;
// 一辆车若上一帧存在装甲板，且这一帧的 A
// 装甲板与上一帧中最近的同车装甲板超过该数值，则抛弃 A
const double NEW_ARMOR_TO_EXISTING_DIS_LIM = 1.2;

EnemyState::EnemyState(const int& number, const aimer::EnemyType& init_type):
    number(number),
    enemy_type(init_type) {}

void EnemyState::update(
    const std::vector<aimer::ArmorData>& raw_armors,
    const double& t,
    const int& frame
) {
    // 排除不可能的装甲板（太远，太小）
    std::vector<aimer::ArmorData> armors = this->screened_armors(raw_armors, this->last_armors);
    if (this->get_enemy_type() == EnemyType::OUTPOST) {
        armors = this->outpost_fixer.filter(armors, t);
    }

    this->last_armors = armors; // maybe empty

    // 非平衡步兵判断机更新
    this->balance_judger.update(armors, t, frame);
    if (this->number >= 3 && this->number <= 5) {
        // 修正车辆类型（目前仅可能修复平衡步兵和非平衡步兵）
        // 若不开启手动设置型号，则进行自动识别
        if (!base::get_param<bool>("auto-aim.enemy-state.manually-set-balance-infantry")) {
            aimer::EnemyType enemy_type_to_be = this->balance_judger.is_balance()
                ? aimer::EnemyType::BALANCE_INFANTRY
                : aimer::EnemyType::INFANTRY;
            this->switch_enemy_type(enemy_type_to_be);
        } else {
            // 根据参数切换敌人类型
            std::string param_str = std::string("auto-aim.enemy-state.infantry")
                + std::to_string(this->number) + std::string("-is-balance");
            aimer::EnemyType enemy_type_to_be = base::get_param<bool>(param_str)
                ? aimer::EnemyType::BALANCE_INFANTRY
                : aimer::EnemyType::INFANTRY;
            this->switch_enemy_type(enemy_type_to_be);
        }
    }
    // 根据车辆类型生成最终数据 conf & dis_center
    this->armor_data = this->sorted_armors(armors, this->get_max_seen());
}

// aimer::EnemyType get_enemy_type() const { return this->enemy_type; }
aimer::EnemyType EnemyState::get_enemy_type() const {
    return this->enemy_type;
}

const aimer::EnemyInfo& EnemyState::get_info_ref() const {
    return aimer::ENEMY_INFO.at(this->enemy_type);
}

int EnemyState::get_armors_num() const {
    return this->get_info_ref().armors_num;
}

int EnemyState::get_max_seen() const {
    return this->get_info_ref().seen_armors_max_num;
}

int EnemyState::get_number() const {
    return this->number;
}

// 为什么用字符串索引，因为要读取参数表的实时更新
double EnemyState::get_aim_error() const { // 调用先远后近，明显不好
    return base::get_param<double>(this->get_info_ref().aim_error_param);
}

double EnemyState::get_default_radius() const {
    return this->get_info_ref().default_radius;
}

// sample 和 pitch 确实分立地属于一种兵种，pitch 不属于 sample，
// sample 是一块没有贴数字的放在桌面上的装甲板
double EnemyState::get_armor_pitch() const {
    return this->get_info_ref().armor_pitch;
}

const aimer::SampleArmor& EnemyState::get_sample_armor_ref() const {
    return this->get_info_ref().sample_armor;
}

const std::vector<aimer::ArmorData>& EnemyState::get_armor_data_ref() const {
    return this->armor_data;
}

bool EnemyState::is_hit() const {
    const std::vector<aimer::ArmorData>& armors = this->get_armor_data_ref();
    bool is_hit = false;
    for (const auto& armor: armors) {
        if (armor.is_hit()) {
            is_hit = true;
        }
    }
    return is_hit;
}

void EnemyState::switch_enemy_type(const aimer::EnemyType& enemy_type) {
    this->enemy_type = enemy_type;
}

// 排除看起来不可能的装甲板
std::vector<aimer::ArmorData> EnemyState::screened_armors(
    const std::vector<aimer::ArmorData>& raw_armors,
    const std::vector<aimer::ArmorData>& last_armors
) {
    std::vector<aimer::ArmorData> armors = raw_armors;
    // vector 中的最大面积
    double max_area = 0.;
    for (auto& d: armors) {
        if (d.info.area() > max_area) {
            max_area = d.info.area();
        }
    }
    for (auto it = armors.begin(); it != armors.end();) {
        bool found_id = false;
        // id 在上一采样点的装甲板中最近的距离
        double closest = aimer::INF;
        for (const auto& d: last_armors) {
            // 需要在活跃的里面取旧的比较么？
            // 哦这里 last 的数据就是活跃的
            if (it->id == d.id) {
                found_id = true;
            }
            double dis = (d.info.pos - it->info.pos).norm();
            if (dis < closest) {
                closest = dis;
            }
        }
        if (!last_armors.empty() && closest > aimer::NEW_ARMOR_TO_EXISTING_DIS_LIM) {
            it = armors.erase(it); // 识别到超远处超小的自然不要
            continue;
        }
        if (found_id) {
            if (it->info.area() < max_area * aimer::EXISTING_ARMOR_CRED_AREA_RATIO) {
                it = armors.erase(it);
                continue;
            }
        } else { // id not found
            if (it->info.area() < max_area * aimer::NEW_ARMOR_CRED_AREA_RATIO
                || it->info.pos.norm()
                    > base::get_param<double>("auto-aim.enemy-state.new-armor-max-valid-distance"))
            {
                it = armors.erase(it);
                continue;
            }
        }
        it++;
    }
    return armors;
}

// 根据指定 size 裁剪和排序装甲板
std::vector<aimer::ArmorData>
EnemyState::sorted_armors(const std::vector<aimer::ArmorData>& src, const int& max_size) {
    std::vector<aimer::ArmorData> res = src;
    std::sort(res.begin(), res.end(), [](const aimer::ArmorData& d1, const aimer::ArmorData& d2) {
        return d1.info.detected.conf > d2.info.detected.conf;
    });
    if (res.size() > static_cast<std::size_t>(max_size)) {
        res = std::vector<aimer::ArmorData>(res.begin(), res.begin() + max_size);
    }
    std::sort(res.begin(), res.end(), [](const aimer::ArmorData& d1, const aimer::ArmorData& d2) {
        return d1.info.center().x < d2.info.center().x;
    }); // 从左往右排序，且容量一定是 1 或者 2
    return res;
}
} // namespace aimer
