#include "aimer/auto_aim/predictor/enemy/armor_identifier.hpp"

#include "aimer/base/debug/debug.hpp"

namespace aimer {

// 单个 light 未捕获数据时最多保留的时间
// （注意，若存留太久有可能错误捕获新出现的）
const double LIGHT_LIFE = 0.100;

const int ENEMY_MAX_LIGHT_CNT = 3;

/** @class LightThread */

LightThread::LightThread(
    const int& id,
    const aimer::ArmorInfo& armor_info,
    const double& t,
    const int& frame
):
    id(id),
    armor_info(armor_info),
    target_color(armor_info.detected.color),
    close_target_t(t),
    frame(frame) {}

// 输出带有序号并修正颜色的装甲板数据
aimer::ArmorData LightThread::get_armor() const {
    return aimer::ArmorData(this->id, this->target_color, this->armor_info);
}

aimer::ArmorInfo LightThread::get_armor_info() const {
    return this->armor_info;
}

bool LightThread::active(const int& frame) const {
    // 这里没问题，target_color 是这个线程的目标 color
    return frame == this->frame;
}

bool LightThread::alive(const ::ArmorColor& enemy_color, const double& t) const {
    return this->target_color == static_cast<int>(enemy_color)
        && t - this->close_target_t <= aimer::LIGHT_LIFE;
}

void LightThread::update(const aimer::ArmorInfo& armor_info, const double& t, const int& frame) {
    this->armor_info = armor_info;
    this->frame = frame;
    if (armor_info.detected.color == this->target_color) {
        this->close_target_t = t;
    }
}

// 把两个装甲板平铺，以对角线为直径的○相切时，代价为 1.0
double LightThread::get_cost(const aimer::ArmorInfo& armor_info) const {
    double cost = 0.;
    if (armor_info.detected.number != this->armor_info.detected.number) {
        cost += 1.1;
    }
    aimer::math::YpdCoord in_ypd = aimer::math::xyz_to_ypd(this->armor_info.pos);
    aimer::math::YpdCoord out_ypd = aimer::math::xyz_to_ypd(armor_info.pos);
    Eigen::Vector3d in_pos = this->armor_info.pos;
    // 消除 dis 过大误差的影响
    Eigen::Vector3d out_pos =
        aimer::math::ypd_to_xyz(aimer::math::YpdCoord(out_ypd.yaw, out_ypd.pitch, in_ypd.dis));
    // 对角线长度的一半为 0.5
    cost += (in_pos - out_pos).norm()
        / (aimer::math::get_norm(this->armor_info.sample.width, this->armor_info.sample.height) / 2.
        )
        * 0.5;
    return cost;
}

bool LightThread::collide(const LightThread& guest) const {
    // 注意此处价值逻辑需要与 id 暂存冲突的逻辑协调
    // 距离过远时，会否因为枪口抖动造成跳 id？
    // 边缘上代价为 0.5（收纳阈值）。两圆相撞，单方面代价为 1，总代价 2.
    return this->get_cost(guest.get_armor_info()) + guest.get_cost(this->get_armor_info()) < 2.;
}

/** @class LightManager */

void LightManager::update(const std::vector<aimer::ArmorInfo>& armor_infos) {
    this->frame = this->converter->get_frame();
    std::deque<aimer::ArmorInfo> info_list;
    for (const auto& a: armor_infos) {
        info_list.push_back(a);
    }
    for (auto& light: this->lights) {
        // light 找 armor，找到后删除 armor，由于未来可能修正 number，复杂度为 n ^ 2
        auto found = info_list.end();
        double min_cost = 0.;
        for (auto it = info_list.begin(); it != info_list.end(); ++it) {
            double cost = light.second.get_cost(*it);
            if (cost > 0.5) {
                continue;
            }
            if (found == info_list.end() || cost < min_cost) {
                found = it;
                min_cost = cost;
            }
        }
        if (found != info_list.end()) {
            light.second.update(*found, this->converter->get_img_t(), this->frame);
            // 数据被捕获后删除该数据
            info_list.erase(found);
        }
    }

    // 对未被捕获的装甲板试图创建新线程
    for (auto& d: info_list) { // 用来更新的 armors 已经删了
        if (d.detected.color
            == static_cast<int>(this->converter->get_robot_status_ref().enemy_color))
        {
            ++this->light_id_cnt;
            this->lights.insert(std::make_pair(
                this->light_id_cnt,
                LightThread(this->light_id_cnt, d, this->converter->get_img_t(), this->frame)
            ));
        }
    }

    // 检查线程是否应删除
    for (auto it = this->lights.begin(); it != this->lights.end();) {
        if (!it->second.alive(
                static_cast<::ArmorColor>(this->converter->get_robot_status_ref().enemy_color),
                this->converter->get_img_t()
            ))
        {
            it = this->lights.erase(it);
        } else {
            ++it;
        }
    }
    // 也可以对 number 限制为 max_seen，但是目前不太好获取 max_seen 和数量
    // 不知板子数量就无法确定类型。不能用当前类型反向决定板子数量
    // 可以对 number 限制最大数量，否则 n ^ 2 吃不消
    {
        int light_cnt[aimer::MAX_ENEMY_NUMBER + 1] = { 0 };
        for (auto& d: this->lights) {
            light_cnt[d.second.get_armor_info().detected.number] += 1;
        }
        for (auto it = this->lights.begin(); it != this->lights.end();) {
            if (!it->second.active(this->frame)) {
                // 如果你真的 active，那我不砍你。如果你是休眠状态，那我只能砍掉你了
                int number = it->second.get_armor_info().detected.number;
                bool collide = false;
                for (auto& d: lights) {
                    if (it->second.get_id() == d.second.get_id()) {
                        continue; // 设定自己和自己不碰撞
                    }
                    if (it->second.collide(d.second)) {
                        collide = true;
                    }
                }
                if (collide || light_cnt[number] > aimer::ENEMY_MAX_LIGHT_CNT) {
                    light_cnt[number] -= 1;
                    it = this->lights.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }

    // std::cout << "[PRED] LIGHT threads count: " << this->lights.size() << '\n';
}

std::vector<aimer::ArmorData> LightManager::get_all_armors() const {
    std::vector<aimer::ArmorData> res;
    // 这边输出本帧看见的装甲板
    for (const auto& d: this->lights) {
        if (d.second.active(this->frame)) {
            res.push_back(d.second.get_armor());
        }
    }
    return res;
}

} // namespace aimer
