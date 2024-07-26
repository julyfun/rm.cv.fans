#ifndef AIMER_AUTO_AIM_PREDICTOR_ENEMY_ARMOR_IDENTIFIER_HPP
#define AIMER_AUTO_AIM_PREDICTOR_ENEMY_ARMOR_IDENTIFIER_HPP

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"

// 装甲板标序号和熄灭补偿方法类，与 aimer::ArmorData 耦合存在。
//  - 在线程中找不到够近的：++id_cnt, 新增 light 线程
//  - 在线程中找到够近的：更新线程
//  - 删除太久未被目标颜色更新的（被非目标颜色更新时，
//    update_t 不会跟进）
namespace aimer {
class LightThread {
public:
    LightThread(
        const int& id,
        const aimer::ArmorInfo& armor_info,
        const double& t,
        const int& frame);

    int get_id() const {
        return this->id;
    }
    aimer::ArmorInfo get_armor_info() const;
    // 输出带有序号并修正颜色的装甲板数据
    aimer::ArmorData get_armor() const;

    // 在当前帧是否获取数据
    bool active(const int& frame) const;

    // 是否存活
    bool alive(const ::ArmorColor& enemy_color, const double& t) const;

    // （如果认为是同一块装甲板）更新 Light 线程信息
    void update(const aimer::ArmorInfo& armor_info, const double& t, const int& frame);

    // 计算捕获代价
    double get_cost(const aimer::ArmorInfo& info) const;

    // 判断是否与另一线程碰撞，如果本线程休眠且与其他任一线程碰撞，则删除本线程
    bool collide(const aimer::LightThread& guest) const;

private:
    int id; // 本线程 id（固定）
    aimer::ArmorInfo armor_info; // 当前的装甲板数据
    int target_color; // 本线程颜色（固定）
    double close_target_t; // 最近一次颜色正确的时间（用于颜色修正）
    int frame;
    /*  当同一 id
     的装甲板颜色错误（主要是因为被子弹打击后，熄灭0.05s，亮0.05s，再熄灭0.05s）后，
      在*这段时间*内认为它是原来那块
      神经网络能识别熄灭的装甲板，但颜色随机*/
};

// 对装甲板线程进行管理
class LightManager {
public:
    explicit LightManager(aimer::CoordConverter* const converter): converter(converter) {}
    void update(const std::vector<aimer::ArmorInfo>& armor_infos);

    std::vector<aimer::ArmorData> get_all_armors() const;

private:
    aimer::CoordConverter* const converter;
    // 装甲板线程，用于熄灭装甲板补全和 id 识别
    std::unordered_map<int, aimer::LightThread> lights;
    int frame;
    int light_id_cnt = 0; // 装甲板线程计数
};
} // namespace aimer

#endif /* AIMER_AUTO_AIM_PREDICTOR_ENEMY_ARMOR_IDENTIFIER_HPP */
