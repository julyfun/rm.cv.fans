aimer::AimInfo TopModel::get_aim(
    aimer::CoordConverter* const converter,
    aimer::EnemyState* const enemy_state
) const {
    using ::base::get_param;
    if (!this->credit(converter)) {
        return aimer::AimInfo::idle();
    }
    // todo
    const int armors_num = enemy_state->get_armors_num();
    std::vector<ArmorFilter> filters = this->get_armor_filters(enemy_state);
    // ... num
    std::vector<State> water_gun_hit_state_vec = {};
    for (const auto& filter: filters) {
        const double water_gun_hit_time = converter->filter_to_prediction_time(filter);
        const State that_state = filter.predict(water_gun_hit_time);
        water_gun_hit_state_vec.emplace_back(that_state);
    }
    // [选择应该命令电机转向哪个装甲板]
    // 如果有 angle 以内的，直接选择其中需要转动角最小的那个就行
    // 否则需要计算 emerging pos（即将出现在合适打击的位置）
    const double max_orientation_angle = aimer::math::deg_to_rad([&]() {
        if (this->top_level == 0) {
            if (armors_num == 4) {
                return get_param<double>(
                    "auto-aim.lmtd-top-model.aim.top0.max-orientation-angle.armors-4"
                );
            }
            return get_param<double>(
                "auto-aim.lmtd-top-model.aim.top0.max-orientation-angle.armors-other"
            );
        }
        return get_param<double>(
            fmt::format("auto-aim.lmtd-top-model.aim.top{}.max-orientation-angle", this->top_level)
        );
    }());
    const double max_out_error = get_param<double>(
        fmt::format("auto-aim.lmtd-top-model.aim.top{}.max-out-error", this->top_level)
    );
    const bool allow_indirect = this->top_level > 0;
    const auto water_gun_hit_aim_and_state = state_vec_to_aim(
        water_gun_hit_state_vec,
        max_orientation_angle,
        max_out_error,
        allow_indirect,
        converter,
        enemy_state
    );
    if (water_gun_hit_aim_and_state.aim.shoot == ::ShootMode::IDLE) {
        return aimer::AimInfo::idle();
    }
    std::vector<State> command_hit_state_vec = {};
    for (const auto& filter: filters) {
        const double command_hit_time = converter->filter_to_hit_time(filter);
        const State that_state = filter.predict(command_hit_time);
        command_hit_state_vec.emplace_back(that_state);
    }
    const auto command_hit_aim_and_state = state_vec_to_aim(
        command_hit_state_vec,
        max_orientation_angle,
        max_out_error,
        /*allow_indirect=*/true,
        converter,
        enemy_state
    );
    const bool you_had_better_shoot_at_this_command = [&]() {
        // [如果跟随误差太大就不发]
        const double max_tracking_error =
            ::base::get_param<double>("auto-aim.lmtd-top-model.aim.max-tracking-error");
        if (converter->aim_error_exceeded(
                water_gun_hit_aim_and_state.aim.ypd,
                enemy_state->get_sample_armor_ref(),
                max_tracking_error,
                state_to_zn_to_armor(water_gun_hit_aim_and_state.state, converter),
                enemy_state->get_armor_pitch()
            ))
        {
            return false; // don't send shoot command
        }
        // [如果 command_hit 的时候在旋转枪口那就不能发]
        // 首先判断 water_gun_hit 到 command_hit 是否在回转
        // 如果是回转的，可以算出开始回转的时间点
        // 也可以算出回转的角度有多大（有一个近似，就是假设转到 command_hit 的角度）
        // 然后根据 angle_to_rotate_time 算出回转结束的时间点
        // 如果 command_hit time 在两个之间就不能发
        // [.注意] 区分枪口的 yaw 和装甲板的 yaw。两个在不同地方使用

        // 在 top_level == 0 时角速度不可信，这里不能用来算关键时间点
        if (this->top_level > 0) {
            const auto angle_to_rotate_time = [](const double& angle) -> double {
                const double a =
                    base::get_param<double>("auto-aim.lmtd-top-model.aim.angle-to-rotate-time.a");
                const double b =
                    base::get_param<double>("auto-aim.lmtd-top-model.aim.angle-to-rotate-time.b");
                // 注意这里线性函数的参数是角度制哦
                return a * aimer::math::rad_to_deg(angle) + b;
            };
            const double w_water_gun_hit = water_gun_hit_aim_and_state.state[7];
            const double armor_rotate_water_gun_hit_to_command_hit = aimer::math::reduced_angle(
                command_hit_aim_and_state.state[6] - water_gun_hit_aim_and_state.state[6]
            );

            // 这段逻辑很复杂但是看起来运行的还挺正常
            // base::print_info(
            //     "---\nw: {}\nr: {}",
            //     aimer::math::rad_to_deg(w_water_gun_hit),
            //     aimer::math::rad_to_deg(armor_rotate_water_gun_hit_to_command_hit)
            // );

            if (std::signbit(w_water_gun_hit)
                != std::signbit(armor_rotate_water_gun_hit_to_command_hit))
            {
                // 小心，这里没有预防 water_gun_hit 装甲板超过 max_orientation_angle 的情况
                const double zn_to_armor_water_gun_hit =
                    state_to_zn_to_armor(water_gun_hit_aim_and_state.state, converter);
                // const double zn_to_armor_command_hit =
                //     state_to_zn_to_armor(command_hit_aim_and_state.state, converter);
                const double zn_to_where_you_should_rotate_back =
                    w_water_gun_hit > 0.0 ? +max_orientation_angle : -max_orientation_angle;
                const double armor_water_gun_hit_to_rotate_back = aimer::math::reduced_angle(
                    zn_to_where_you_should_rotate_back - zn_to_armor_water_gun_hit
                );
                // const double angle_rotate_back = aimer::math::reduced_angle(
                //     zn_to_armor_command_hit - zn_to_where_you_should_rotate_back
                // );

                // 这里的 time 是正常原点的时间
                const double time_water_gun_hit = converter->get_prediction_time(
                    state_to_armor_pos(water_gun_hit_aim_and_state.state)
                );
                const double time_command_hit =
                    converter->get_hit_time(state_to_armor_pos(command_hit_aim_and_state.state));
                const double time_start_rotating_back =
                    time_water_gun_hit + armor_water_gun_hit_to_rotate_back / w_water_gun_hit;
                const auto filter =
                    ArmorFilter { water_gun_hit_aim_and_state.state, time_water_gun_hit };
                const auto pos_when_start_rotating_back =
                    filter.predict_pos(time_start_rotating_back);
                const auto aim_when_start_rotating_back =
                    converter->target_pos_to_aim_ypd(pos_when_start_rotating_back);
                const double yaw_barrel_rotate_back = aimer::math::reduced_angle(
                    command_hit_aim_and_state.aim.ypd.yaw - aim_when_start_rotating_back.yaw
                );
                const double time_end_rotating_back = time_start_rotating_back
                    + angle_to_rotate_time(std::abs(yaw_barrel_rotate_back));
                // base::print_info(
                //     "***\nrotate: {} {}",
                //     yaw_barrel_rotate_back,
                //     angle_to_rotate_time(std::abs(yaw_barrel_rotate_back))
                // );
                // 完蛋了我不知道 command_hit_time
                // 无所谓，原地算好了

                // base::print_info(
                //     "***\ns: {}\ne: {}\nh: {}",
                //     time_start_rotating_back,
                //     time_end_rotating_back,
                //     time_command_hit
                // );
                if (time_start_rotating_back < time_command_hit
                    && time_command_hit < time_end_rotating_back)
                {
                    return false; // don't send shoot command
                }
            }
        }
        if (converter->aim_error_exceeded(
                command_hit_aim_and_state.aim.ypd,
                converter->target_pos_to_aim_ypd(state_to_armor_pos(command_hit_aim_and_state.state)
                ),
                enemy_state->get_sample_armor_ref(),
                max_out_error,
                state_to_zn_to_armor(command_hit_aim_and_state.state, converter),
                enemy_state->get_armor_pitch()
            ))
        {
            return false; // don't send shoot command
        }
        return true; // send shoot command
    }();
    auto aim_with_shoot_command = water_gun_hit_aim_and_state.aim;
    aim_with_shoot_command.shoot =
        you_had_better_shoot_at_this_command ? ::ShootMode::SHOOT_NOW : ::ShootMode::TRACKING;
    return aim_with_shoot_command;
}
