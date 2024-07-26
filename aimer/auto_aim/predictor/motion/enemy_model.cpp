#include "aimer/auto_aim/predictor/motion/enemy_model.hpp"

#include <fmt/color.h>
#include <fmt/format.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "aimer/auto_aim/predictor/motion/armor_model.hpp"
#include "aimer/auto_aim/predictor/motion/top_model.hpp"
#include "aimer/base/debug/debug.hpp"

namespace aimer {

// 禁止使用 extern 声明跨文件全局变量，否则将导致构造顺序的不确定
const std::unordered_map<aimer::EnemyType, aimer::ModelType> ENEMY_TO_MODEL = {
    { aimer::EnemyType::OLD_SENTRY, aimer::ModelType::SENTRY },
    { aimer::EnemyType::HERO, aimer::ModelType::INFANTRY },
    { aimer::EnemyType::ENGINEER, aimer::ModelType::INFANTRY },
    { aimer::EnemyType::INFANTRY, aimer::ModelType::INFANTRY },
    { aimer::EnemyType::BALANCE_INFANTRY, aimer::ModelType::BALANCE_INFANTRY },
    { aimer::EnemyType::OUTPOST, aimer::ModelType::OUTPOST },
    { aimer::EnemyType::CRYSTAL_BIG, aimer::ModelType::STATUE },
    { aimer::EnemyType::CRYSTAL_SMALL, aimer::ModelType::STATUE },
};

const std::unordered_map<aimer::ModelType, double> MODEL_TOP_CREDIT_TIME = {
    { aimer::ModelType::INFANTRY, 0.1 },
    { aimer::ModelType::OUTPOST, 1. },
    { aimer::ModelType::BALANCE_INFANTRY, 0.5 },
};

// 注意 id 线程生命周期是 0.1s，而滤波器线程的存活则完全取决于 credit
const std::unordered_map<aimer::ModelType, double> MODEL_ARMOR_CREDIT_TIME = {
    { aimer::ModelType::SENTRY, 0.08 },  { aimer::ModelType::INFANTRY, 0.08 },
    { aimer::ModelType::OUTPOST, 0.08 }, { aimer::ModelType::BALANCE_INFANTRY, 0.08 },
    { aimer::ModelType::STATUE, 0.08 },
};

/** @class Sentry */

Sentry::Sentry(aimer::CoordConverter* const converter, aimer::EnemyState* const state):
    converter(converter),
    state(state),
    armor_model(converter, state, aimer::MODEL_ARMOR_CREDIT_TIME.at(aimer::ModelType::SENTRY)) {}

bool Sentry::alive() const {
    return this->get_model_type() == aimer::ENEMY_TO_MODEL.at(this->state->get_enemy_type());
}

void Sentry::update() {
    // 常规运动模块更新
    this->armor_model.update();
}

// send 的 shoot 也需要我处理
aimer::AimInfo Sentry::get_aim() const {
    return this->armor_model.get_aim(/*passive=*/
                                     ::base::get_param<bool>(
                                         "auto-aim.enemy-model.old-sentry.passive-mode.enabled"
                                     )
    );
}

void Sentry::draw_aim(cv::Mat& img) const {
    this->armor_model.draw_aim(img);
}

/** @class Infantry */
// number 是 Enemy 的序号，所以在 Enemy 的外部
// 会陀螺的目标

Infantry::Infantry(aimer::CoordConverter* const converter, aimer::EnemyState* const state):
    converter(converter),
    state(state),
    armor_model(converter, state, aimer::MODEL_ARMOR_CREDIT_TIME.at(aimer::ModelType::INFANTRY)),
    top4_model(converter, state, aimer::MODEL_TOP_CREDIT_TIME.at(aimer::ModelType::INFANTRY)) {}

bool Infantry::alive() const {
    return this->get_model_type() == aimer::ENEMY_TO_MODEL.at(this->state->get_enemy_type());
}

void Infantry::update() {
    this->armor_model.update();
    // this->top4_model.update();
    this->lmtd_top_model.update(this->converter, this->state);
}

// send 的 shoot 也需要我处理
aimer::AimInfo Infantry::get_aim() const {
    if (this->lmtd_top_model.get_top_level() >= 1) {
        return this->lmtd_top_model.get_aim(this->converter, this->state);
    }
    return this->armor_model.get_aim(false);
    // aimer::AimInfo aim = aimer::AimInfo::idle();
    // if (this->top4_model.active()) {
    //     aim = this->top4_model.get_limit_aim();
    //     aim.info |= aimer::AimInfo::TOP; // 反陀螺激活
    // } else {
    //     aim = this->armor_model.get_aim(/*passive=*/false);
    // }
    // return aim;
}

void Infantry::draw_aim(cv::Mat& img) const {
    // this->armor_model.draw_aim(img);
    // this->top4_model.draw_aim(img);
    // if (this->top4_model.active()) {
    // aimer::debug::flask_aim << fmt::format("{}: Top Act", this->state->get_number());
    // }
    this->lmtd_top_model
        .draw_armors(img, this->converter->get_img_t(), this->converter, this->state);
}

/** @class BalanceInfantry */

BalanceInfantry::BalanceInfantry(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state
):
    converter(converter),
    state(state),
    // C++ 的规则真是一团糟！
    armor_model { aimer::ArmorModel(
        converter,
        state,
        aimer::MODEL_ARMOR_CREDIT_TIME.at(aimer::ModelType::BALANCE_INFANTRY)
    ) },
    top2_model { aimer::top::SimpleTopModel(
        /*converter=*/converter,
        /*state=*/state,
        /*armor_cnt=*/2,
        /*jump_angle=*/M_PI / 8.,
        /*min_active_rotate*/ 2,
        /*credit_time=*/
        aimer::MODEL_TOP_CREDIT_TIME.at(aimer::ModelType::BALANCE_INFANTRY),
        /*cons=*/ { 0., 0., 0. }
    ) } {}

bool BalanceInfantry::alive() const {
    return this->get_model_type() == aimer::ENEMY_TO_MODEL.at(this->state->get_enemy_type());
}

void BalanceInfantry::update() {
    this->armor_model.update();
    // this->top2_model.update(false);
    this->lmtd_top_model.update(this->converter, this->state);
}

aimer::AimInfo BalanceInfantry::get_aim() const {
    if (this->lmtd_top_model.get_top_level() >= 1) {
        return this->lmtd_top_model.get_aim(this->converter, this->state);
    }
    return this->armor_model.get_aim(false);
    // aimer::AimInfo aim = aimer::AimInfo::idle();
    // if (this->top4_model.active()) {
    //     aim = this->top4_model.get_limit_aim();
    //     aim.info |= aimer::AimInfo::TOP; // 反陀螺激活
    // } else {
    //     aim = this->armor_model.get_aim(/*passive=*/false);
    // }
    // return aim;
}

void BalanceInfantry::draw_aim(cv::Mat& img) const {
    // this->armor_model.draw_aim(img);
    // this->top2_model.draw_aim(img);
    // if (this->top2_model.active()) {
    //     aimer::debug::flask_aim << fmt::format("{}: Top Act", this->state->get_number());
    // }
    this->lmtd_top_model
        .draw_armors(img, this->converter->get_img_t(), this->converter, this->state);
}

/** @class Outpost */

Outpost::Outpost(aimer::CoordConverter* const converter, aimer::EnemyState* const state):
    converter(converter),
    state(state),
    armor_model(converter, state, aimer::MODEL_ARMOR_CREDIT_TIME.at(aimer::ModelType::OUTPOST)),
    top3_model(
        converter,
        state,
        /*armor_cnt=*/3,
        /*jump_angle=*/M_PI / 8.,
        /*min_active_rotate=*/1,
        aimer::MODEL_TOP_CREDIT_TIME.at(aimer::ModelType::OUTPOST),
        aimer::top::OrientationSignFixerConstructor(2., 0.1, M_PI / 8.)
    ) {}

bool Outpost::alive() const {
    return this->get_model_type() == aimer::ENEMY_TO_MODEL.at(this->state->get_enemy_type());
}

void Outpost::update() {
    this->armor_model.update();
    // this->top3_model.update(
    // base::get_param<bool>("auto-aim.enemy-model.outpost.top-orientation-fixer.enabled")
    // );
    this->lmtd_top_model.update(this->converter, this->state);
}

// send 的 shoot 也需要我处理
aimer::AimInfo Outpost::get_aim() const {
    if (this->lmtd_top_model.get_top_level() >= 1) {
        return this->lmtd_top_model.get_aim(this->converter, this->state);
    }
    return this->armor_model.get_aim(false);
    // aimer::AimInfo aim = aimer::AimInfo::idle();
    // if (this->top4_model.active()) {
    //     aim = this->top4_model.get_limit_aim();
    //     aim.info |= aimer::AimInfo::TOP; // 反陀螺激活
    // } else {
    //     aim = this->armor_model.get_aim(/*passive=*/false);
    // }
    // return aim;
}

void Outpost::draw_aim(cv::Mat& img) const {
    // this->armor_model.draw_aim(img);
    // this->top3_model.draw_aim(img);
    // if (this->top3_model.active()) {
    //     aimer::debug::flask_aim << fmt::format("{}: Top Act", this->state->get_number());
    // }
    this->lmtd_top_model
        .draw_armors(img, this->converter->get_img_t(), this->converter, this->state);
}

/** @class Statue */

Statue::Statue(aimer::CoordConverter* const converter, aimer::EnemyState* const state):
    converter(converter),
    state(state),
    armor_model(converter, state, aimer::MODEL_ARMOR_CREDIT_TIME.at(aimer::ModelType::STATUE)) {}

bool Statue::alive() const {
    return this->get_model_type() == aimer::ENEMY_TO_MODEL.at(this->state->get_enemy_type());
}

void Statue::update() {
    // 常规运动模块更新
    this->armor_model.update();
}

// send 的 shoot 也需要我处理
aimer::AimInfo Statue::get_aim() const {
    aimer::AimInfo aim = this->armor_model.get_aim(/*passive=*/false); // converter means now
    return aim;
}

void Statue::draw_aim(cv::Mat& img) const {
    this->armor_model.draw_aim(img);
}

const std::unordered_map<
    aimer::ModelType,
    std::function<std::unique_ptr<aimer::EnemyModelInterface>(
        aimer::CoordConverter* const converter,
        aimer::EnemyState* const state
    )>>
    MODEL_MAP = { { aimer::ModelType::SENTRY,
                    [](aimer::CoordConverter* const converter, aimer::EnemyState* const state) {
                        return std::make_unique<Sentry>(converter, state);
                    } },
                  { aimer::ModelType::INFANTRY,
                    [](aimer::CoordConverter* const converter, aimer::EnemyState* const state) {
                        return std::make_unique<Infantry>(converter, state);
                    } },
                  { aimer::ModelType::BALANCE_INFANTRY,
                    [](aimer::CoordConverter* const converter, aimer::EnemyState* const state) {
                        return std::make_unique<BalanceInfantry>(converter, state);
                    } },
                  { aimer::ModelType::OUTPOST,
                    [](aimer::CoordConverter* const converter, aimer::EnemyState* const state) {
                        return std::make_unique<Outpost>(converter, state);
                    } },
                  { aimer::ModelType::STATUE,
                    [](aimer::CoordConverter* const converter, aimer::EnemyState* const state) {
                        return std::make_unique<Statue>(converter, state);
                    } } };

/** @class EnemyModelFactory */

std::unique_ptr<aimer::EnemyModelInterface> EnemyModelFactory::create_model(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const state
) {
    return aimer::MODEL_MAP.at(aimer::ENEMY_TO_MODEL.at(state->get_enemy_type()))(converter, state);
}
} // namespace aimer
