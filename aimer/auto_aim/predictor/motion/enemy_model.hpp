/*敌方预测模块*/

#ifndef AIMER_AUTO_AIM_PREDICTOR_MOTION_ENEMY_MODEL_HPP
#define AIMER_AUTO_AIM_PREDICTOR_MOTION_ENEMY_MODEL_HPP

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/predictor/enemy/enemy_state.hpp"
#include "aimer/auto_aim/predictor/motion/armor_model.hpp"
#include "aimer/auto_aim/predictor/motion/lmtd_top_model.hpp"
#include "aimer/auto_aim/predictor/motion/top4_model.hpp"
#include "aimer/auto_aim/predictor/motion/top_model.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/math/math.hpp"
#include "aimer/base/robot/coord_converter.hpp"
#include "core_io/robot.hpp"

namespace aimer {
// 最好不要在 enemy_model.xpp 之外使用该 enum class
enum class ModelType { SENTRY, INFANTRY, BALANCE_INFANTRY, OUTPOST, STATUE };

class EnemyModelInterface {
public:
    virtual ~EnemyModelInterface() = default;
    virtual bool alive() const = 0;
    virtual void update() = 0; // 更新信息
    // 返回目标的控制 ypd 与 shoot信息
    virtual aimer::AimInfo get_aim() const = 0;
    virtual void draw_aim(cv::Mat&) const = 0;
};

class Sentry: public aimer::EnemyModelInterface {
public:
    // floTODO 编译之前填完
    Sentry(aimer::CoordConverter* const converter, aimer::EnemyState* const state);
    ~Sentry() override = default;
    bool alive() const override;
    void update() override;
    aimer::AimInfo get_aim() const override;
    void draw_aim(cv::Mat&) const override;

private:
    aimer::ModelType get_model_type() const {
        return aimer::ModelType::SENTRY;
    }
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    aimer::ArmorModel armor_model;
};

// number 是 Enemy 的序号，所以在 Enemy 的外部
// 会陀螺的目标
class Infantry: public aimer::EnemyModelInterface {
public:
    Infantry(aimer::CoordConverter* const converter, aimer::EnemyState* const state);
    // aim_error_rate 也可以传给 top_model
    ~Infantry() override = default;
    bool alive() const override;
    void update() override; // 访问数据库更新信息
    aimer::AimInfo get_aim() const override; // 做出打击决策，返回目标 ypd
    void draw_aim(cv::Mat&) const override;

private:
    aimer::ModelType get_model_type() const {
        return aimer::ModelType::INFANTRY;
    }
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    aimer::ArmorModel armor_model;
    aimer::top::top4::TopModel top4_model;
    aimer::lmtd::TopModel lmtd_top_model;
};

// 平衡步兵
class BalanceInfantry: public aimer::EnemyModelInterface {
public:
    BalanceInfantry(aimer::CoordConverter* const converter, aimer::EnemyState* const state);
    ~BalanceInfantry() override = default;
    bool alive() const override;
    void update() override; // 访问数据库更新信息
    aimer::AimInfo get_aim() const override; // 做出打击决策，返回目标 ypd
    void draw_aim(cv::Mat&) const override;

private:
    aimer::ModelType get_model_type() const {
        return aimer::ModelType::BALANCE_INFANTRY;
    }
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    aimer::ArmorModel armor_model;
    aimer::top::SimpleTopModel top2_model;
    aimer::lmtd::TopModel lmtd_top_model;
};

// 前哨站
class Outpost: public aimer::EnemyModelInterface {
public:
    Outpost(aimer::CoordConverter* const converter, aimer::EnemyState* const state);
    ~Outpost() override = default;
    bool alive() const override;
    void update() override; // 访问数据库更新信息
    aimer::AimInfo get_aim() const override; // 做出打击决策，返回目标 ypd
    void draw_aim(cv::Mat&) const override;

private:
    aimer::ModelType get_model_type() const {
        return aimer::ModelType::OUTPOST;
    }
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    aimer::ArmorModel armor_model;
    top::SimpleTopModel top3_model;
    aimer::lmtd::TopModel lmtd_top_model;
};

// 静止建筑，水晶
class Statue: public aimer::EnemyModelInterface {
public:
    // floTODO 编译之前填完
    Statue(aimer::CoordConverter* const converter, aimer::EnemyState* const state);
    ~Statue() override = default;
    bool alive() const override;
    void update() override;
    aimer::AimInfo get_aim() const override;
    void draw_aim(cv::Mat&) const override;

private:
    aimer::ModelType get_model_type() const {
        return aimer::ModelType::STATUE;
    }
    aimer::CoordConverter* const converter;
    aimer::EnemyState* const state;
    aimer::ArmorModel armor_model;
};

class EnemyModelFactory {
public:
    std::unique_ptr<aimer::EnemyModelInterface>
    create_model(aimer::CoordConverter* const converter, aimer::EnemyState* const state);

private:
};
} // namespace aimer

#endif /* AIMER_AUTO_AIM_PREDICTOR_MOTION_ENEMY_MODEL_HPP */
