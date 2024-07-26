#include <fmt/color.h>
#include <fmt/format.h>
#include <pybind11/numpy.h>

#include <chrono>
#include <thread>

#include "UltraMultiThread/include/umt/ObjManager.hpp"
#include "UltraMultiThread/include/umt/umt.hpp"
#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/predictor/enemy_predictor/enemy_predictor.hpp"
#include "aimer/base/armor_defs.hpp"
#include "base/debug/debug.hpp"
#include "base/param/parameter.hpp"
#include "common/common.hpp"
#include "core_io/robot.hpp"

namespace aimer {

namespace umt = ::umt;
namespace base = ::base;
using namespace std::chrono_literals; // like 200ms

void predictor_run() {
    // 订阅器，接收来自 detector 线程的识别数据
    auto subscriber = umt::Subscriber<aimer::DetectionResult>("detections");
    // 接收电控数据
    auto robot_status = umt::ObjManager<::RobotStatus>::find_or_create("robot_status");
    // 发布器，点击控制指令发送者
    auto robot_cmd = umt::Publisher<::RobotCmd>("robot_cmd");

    // 第一视角自瞄信息
    auto predictor_aim_checkbox =
        umt::ObjManager<::CheckBox>::find_or_create("auto_aim.predictor.aim");
    auto webview_predictor_aim = umt::Publisher<cv::Mat>("auto_aim.predictor.aim");

    // 第三视角自瞄信息
    auto predictor_map_checkbox =
        umt::ObjManager<::CheckBox>::find_or_create("auto_aim.predictor.map");
    auto webview_predictor_map = umt::Publisher<cv::Mat>("auto_aim.predictor.map");

    // 网页上显示自瞄数据，而且可以用按钮展开和折叠
    auto webview_info_page = ::base::webview_info_hold("auto_aim.predictor.aim");

    aimer::DetectionResult data = {};
    ::RobotCmd send = {};

    base::wait_for_param("ok");
    base::print(base::PrintMode::INFO, "auto_aim.predictor", "看起来参数创建完了。");
    auto enemy_predictor = std::make_unique<aimer::EnemyPredictor>();

    // auto webview_robot_info = umt::Publisher<cv::Mat> { "终端" };
    auto webview_page_info = umt::ObjManager<::base::webview_info::Page>::find_or_create("root");

    int fps = 0, fps_count = 0;
    auto t1 = std::chrono::system_clock::now();

    while (true) {
        try {
            // MANUAL: 计算但不动枪口且不发射（可发信号来给 UI 反馈）
            // ENERGY相关: 停止计算进程，给能量机关计算
            if (robot_status->program_mode == ::ProgramMode::ENERGY_HIT
                || robot_status->program_mode == ::ProgramMode::ENERGY_DISTURB)
            {
                std::this_thread::sleep_for(200ms);
                continue;
            }
            data = subscriber.pop();
            send = enemy_predictor->predict(data);
            if (std::isnan(send.yaw) || std::isnan(send.pitch) || std::isinf(send.yaw)
                || std::isinf(send.pitch))
            {
                fmt::print(
                    fmt::fg(fmt::color::red),
                    "====================Predictor output nan of inf, rebuilt it======="
                    "=============\n"
                );
                enemy_predictor = nullptr; // 删除
                enemy_predictor = std::make_unique<aimer::EnemyPredictor>();
                std::this_thread::sleep_for(200ms); // 是否需要 sleep？
                continue;
            }
            // if (recv_data->program_mode == ProgramMode::AUTOAIM) {
            //   if (found_target) {
            //     send.seq_id++;
            //     robot_cmd_pub.push(send);
            //     // （信号持续版）自瞄是否在发包？（最好）
            //     // （看到车持续版）看到车了以后自瞄是否在发包？（不管是什么模式）
            //   }
            // }
            send.seq_id++; // 若没有目标，则 predict 函数内 send 未改动
            robot_cmd.push(send);

            {
                fps_count++;
                auto t2 = std::chrono::system_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() >= 1000)
                {
                    fps = fps_count;
                    fps_count = 0;
                    t1 = t2;
                }
                webview_page_info->sub("自瞄-预测器").sub("帧率").get() = fmt::format("{}", fps);
                webview_page_info->sub("自瞄-预测器").sub("开机时间").get() = fmt::format(
                    "{:.3f} 秒",
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    )
                            .count()
                        / 1e3
                );
            }

            if (predictor_aim_checkbox->checked) {
                cv::Mat im2show { enemy_predictor->draw_aim(data.img, data) };
                cv::putText(
                    im2show,
                    fmt::format("fps={}", fps),
                    { 10, 25 },
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    { 0, 0, 255 }
                );
                webview_predictor_aim.push(im2show);
            }
            if (predictor_map_checkbox->checked) {
                cv::Mat img { enemy_predictor->draw_map() };
                webview_predictor_map.push(img);
            }
        } catch (umt::MessageError& e) {
            fmt::print(fmt::fg(fmt::color::orange), "[WARNING] 'detection_data' {}\n", e.what());
            std::this_thread::sleep_for(500ms);
        }
    }
}

void background_predictor_run() {
    std::thread([]() { aimer::predictor_run(); }).detach();
}
} // namespace aimer

PYBIND11_EMBEDDED_MODULE(auto_aim_predictor, m) {
    // namespace py = pybind11;
    m.def("background_predictor_run", aimer::background_predictor_run);
}
