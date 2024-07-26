//
// Created by sjturm on 2020/7/13.
//
// Edited by shenyibo on 2022/6/13. its for opencv

#include <fmt/color.h>
#include <fmt/format.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "UltraMultiThread/include/umt/umt.hpp"
#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/auto_aim/detector_opencv/TRTModule.hpp"
#include "aimer/base/armor_defs.hpp"
#include "common/common.hpp"
#include "core_io/robot.hpp"
#include "core_io/sensors.hpp"

// namespace base
#include "base/debug/debug.hpp"

#define DEAD_GRAY_ARMOR 0
#define LIVED_GRAY_ARMOR 1
bool state = false;
bool lockTargetState = false;
int lockTargetCnt = 0;
int lockTargetIdx = -1;
bool loseTargetFlag = true;
using namespace std::chrono;
namespace py = pybind11;

float thres_partiel_armor[10] = { 0.5f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.35f, 0.35f, 0.35f, 0.0f };
float thres_partiel_energy[10] = { 0.35f, 0.4f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
const Attributes attributes_armor = (struct Attributes) { 4, 4, 9, thres_partiel_armor, true };
const Attributes attributes_energy = (struct Attributes) { 5, 2, 3, thres_partiel_energy, true };

void draw_boxes(cv::Mat& src, const std::vector<bbox_t>& results) {
    cv::Scalar color_map[3] = { { 255, 0, 0 }, { 0, 0, 255 }, { 0, 255, 0 } };
    for (const auto& box: results) {
        const auto& color = color_map[box.color_id];
        cv::line(src, box.pts[0], box.pts[1], color, 1);
        cv::line(src, box.pts[1], box.pts[2], color, 1);
        cv::line(src, box.pts[2], box.pts[3], color, 1);
        cv::line(src, box.pts[3], box.pts[0], color, 1);
        cv::putText(src, std::to_string(box.tag_id), box.pts[0], 0, 1, color);
        cv::putText(src, std::to_string(box.confidence).substr(0, 4), box.pts[2], 0, 1, color);
        cv::putText(
            src,
            "c" + std::to_string(box.confidence_cls).substr(0, 4),
            box.pts[3],
            0,
            1,
            color
        );
    }
}

bool cmp_armor(aimer::DetectedArmor a1, aimer::DetectedArmor a2) {
    auto l1 = sqrt(pow((a1.pts[0].x - a1.pts[3].x), 2) + pow((a1.pts[0].y - a1.pts[3].y), 2));
    auto h1 = sqrt(pow((a1.pts[0].x - a1.pts[1].x), 2) + pow((a1.pts[0].y - a1.pts[1].y), 2));
    auto l2 = sqrt(pow((a2.pts[0].x - a2.pts[3].x), 2) + pow((a2.pts[0].y - a2.pts[3].y), 2));
    auto h2 = sqrt(pow((a2.pts[0].x - a2.pts[1].x), 2) + pow((a2.pts[0].y - a2.pts[1].y), 2));
    return (l1 * h1) > (l2 * h2);
}

// 根据离装甲板中心距离确定
bool cmp_armor_center(aimer::DetectedArmor a1, aimer::DetectedArmor a2) {
    cv::Point2d center = { 640, 384 };
    cv::Point2d center1 = { (a1.pts[0].x + a1.pts[1].x + a1.pts[2].x + a1.pts[3].x) / 4.f,
                            (a1.pts[0].y + a1.pts[1].y + a1.pts[2].y + a1.pts[3].y) / 4.f };
    cv::Point2d center2 = { (a2.pts[0].x + a2.pts[1].x + a2.pts[2].x + a2.pts[3].x) / 4.f,
                            (a2.pts[0].y + a2.pts[1].y + a2.pts[2].y + a2.pts[3].y) / 4.f };
    auto dist1 = sqrt(pow(center1.x - center.x, 2) + pow(center1.y - center.y, 2));
    auto dist2 = sqrt(pow(center2.x - center.x, 2) + pow(center2.y - center.y, 2));
    if (dist1 < dist2) {
        return true;
    } else
        return false;
}

cv::Point2d getBoxCenter(bbox_t b) {
    auto k1 = (b.pts[2].y - b.pts[0].y) / (b.pts[2].x - b.pts[0].x);
    auto b1 = k1 * b.pts[0].x + b.pts[0].y;
    auto k2 = (b.pts[3].y - b.pts[1].y) / (b.pts[3].x - b.pts[1].x);
    auto b2 = k2 * b.pts[3].x + b.pts[3].y;

    double x = (b2 - b1) / (k1 - k2);
    double y = k1 * x + b1;

    return { x, y };
}

bool get_Dist(bbox_t res1, bbox_t res2) {
    double dist = 0;
    double thresh = 100;
    cv::Point2f center_res1, center_res2;

    thresh = 0.5
        * sqrt(pow((res1.pts[0].x - res2.pts[3].x), 2) + pow((res1.pts[0].y - res2.pts[3].y), 2));
    center_res1 = getBoxCenter(res1);
    center_res2 = getBoxCenter(res2);
    dist = sqrt(pow((center_res1.x - center_res2.x), 2) + pow((center_res1.x - center_res2.x), 2));

    if (dist < thresh)
        return true;
    else
        return false;
}

extern cv::Point2d getCenter(aimer::DetectedArmor a) {
    auto k1 = (a.pts[2].y - a.pts[0].y) / (a.pts[2].x - a.pts[0].x);
    auto b1 = k1 * a.pts[0].x + a.pts[0].y;
    auto k2 = (a.pts[3].y - a.pts[1].y) / (a.pts[3].x - a.pts[1].x);
    ;
    auto b2 = k2 * a.pts[3].x + a.pts[3].y;

    double x = (b2 - b1) / (k1 - k2);
    double y = k1 * x + b1;

    return { x, y };
}
bool getTargetArmor(aimer::DetectedArmor a, aimer::DetectedArmor l_a, double& thresh) {
    thresh = sqrt(pow((a.pts[0].x - a.pts[3].x), 2) + pow((a.pts[0].y - a.pts[3].y), 2));
    cv::Point2f center_a = getCenter(a);
    cv::Point2f center_l_a = getCenter(l_a);
    double dist = sqrt(pow((center_a.x - center_l_a.x), 2) + pow((center_a.y - center_l_a.y), 2));
    if (dist < thresh)
        return true;
    else
        return false;
}

void detector_run(const std::string& module_path) {
    // 创建相机陀螺仪数据接收者
    umt::Subscriber<SensorsData> subscriber("sensors_data");
    // 创建识别结果发布者
    umt::Publisher<aimer::DetectionResult> publisher("detections");

    auto detection_checkbox = umt::ObjManager<CheckBox>::find_or_create("auto_aim.detector");
    umt::Publisher<cv::Mat> detections_client("auto_aim.detector");

    auto webview_page_info = umt::ObjManager<::base::webview_info::Page>::find_or_create("root");

    auto recv_data = umt::ObjManager<RobotStatus>::find_or_create("robot_status");
    auto mode_lock = umt::ObjManager<RobotStatus>::find_or_create("mode_lock");
    // 路径加载四点模型

    ::base::print(::base::PrintMode::INFO, "auto_aim.detector", "即将加载神经网络 🚀.");
    SmartModel module(attributes_armor, module_path);

    // TODO:加载Yolo网络模型

    int fps = 0, fps_count = 0;
    auto t1 = system_clock::now();

    ::base::print(::base::PrintMode::INFO, "auto_aim.detector", "即将运行识别主循环.");
    while (true) {
        try {
            if (recv_data->program_mode == ProgramMode::ENERGY_HIT
                || recv_data->program_mode == ProgramMode::ENERGY_DISTURB)
            {
                mode_lock->program_mode = recv_data->program_mode;
                std::this_thread::sleep_for(200ms);
                continue;
            }
            // if (mode_lock->program_mode == ProgramMode::ENERGY_HIT ||
            // mode_lock->program_mode == ProgramMode::ENERGY_DISTURB)
            //{
            //     std::this_thread::sleep_for(200ms);
            //     continue;
            // }
            //  接收传感器数据，但是相机处理传输数据的时间无从得知
            const auto& [image, q_, timestamp] = subscriber.pop();
            // double imu_timestamp = recv_data->imu_timestam
            // 放在 pop 之前或者之后考虑一下
            // 考虑到 sys_fire_time 测量的也是裁判传输过来以后的时间戳，这里也放后面
            // pop 本身几乎不耗时间
            // 均设置为捕获触发记时
            // q转化为四元数
            Eigen::Quaternionf q(q_[0], q_[1], q_[2], q_[3]);

            cv::Mat im2show;

            // 先将四点预测结果存储在tmp变量中
            // auto result_tmp = module(image);
            std::vector<bbox_t> result = module(image);
            std::vector<aimer::DetectedArmor> armors;

            // 进行熄灭装甲板处理
            static int cnt = -1;
            static std::vector<bbox_t> last_outputs;
            static std::vector<aimer::DetectedArmor> last_armors;
            static std::vector<aimer::DetectedArmor> tmp_armors;

            if (cnt == -1) {
                last_outputs = result;
                cnt = 1;
            }

            // 存储上一帧的结果
            last_outputs = result;
            // 对结果信息提取到装甲板
            for (const auto& output: result) {
                //            if (output.tag_id == 2) continue; // 不瞄准工程
                cv::Point2f pts[4] = {
                    output.pts[0],
                    output.pts[1],
                    output.pts[2],
                    output.pts[3],
                };
                aimer::DetectedArmor a_tmp;
                a_tmp.pts[0] = pts[0];
                a_tmp.pts[1] = pts[1];
                a_tmp.pts[2] = pts[2];
                a_tmp.pts[3] = pts[3];
                a_tmp.color = output.color_id;
                a_tmp.number = output.tag_id == 8 ? 6 : output.tag_id;
                // 归并前哨站和水晶小板
                a_tmp.conf = output.confidence;
                a_tmp.conf_class = output.confidence_cls;
                armors.emplace_back(a_tmp);
            }
            // 根据装甲板大小进行sort
            sort(armors.begin(), armors.end(), cmp_armor_center);

            // TODO: 从多个装甲板中选取目标

            // 存储上一次识别的装甲板结果
            last_armors = armors;
            {
                fps_count++;
                auto t2 = system_clock::now();
                if (duration_cast<milliseconds>(t2 - t1).count() >= 1000) {
                    fps = fps_count;
                    fps_count = 0;
                    t1 = t2;
                }
                webview_page_info->sub("自瞄-识别器").sub("帧率").get() = fmt::format("{}", fps);
            }
            // 检测结果绘图&显示
            if (detection_checkbox->checked) {
                im2show = image.clone();
                // im2show = cv::Mat::zeros(cv::Size(1280, 768), CV_8UC1);
                draw_boxes(im2show, result);
                cv::putText(
                    im2show,
                    fmt::format("fps={}", fps),
                    { 10, 25 },
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    { 0, 0, 255 }
                );
                detections_client.push(im2show);
            }

            // 将检测结果放入FIFO
            publisher.push({ image, q, timestamp, armors });

        } catch (umt::MessageError& e) {
            fmt::print(fmt::fg(fmt::color::orange), "[WARNING] 'sensors_data' {}\n", e.what());
            std::this_thread::sleep_for(500ms);
        }
    }
}

void background_detector_run(const std::string& module_path) {
    std::cerr << "=========================background_detector_run==============="
                 "==============="
              << std::endl;
    std::thread([=]() { detector_run(module_path); }).detach();
}

PYBIND11_EMBEDDED_MODULE(auto_aim_detector, m) {
    namespace py = pybind11;
    m.def("background_detector_run", background_detector_run, py::arg("module_path"));
}
