//
// Created by xinyang on 2021/4/28.
//
// Edited by shenyibo on 2022/6/13. its for opencv

#include "TRTModule.hpp"

#include <fmt/color.h>
#include <fmt/format.h>

#include <filesystem>
#include <fstream>

#include "base/debug/debug.hpp"

template<class F, class T, class... Ts>
T reduce(F&& func, T x, Ts... xs) {
    if constexpr (sizeof...(Ts) > 0) {
        return func(x, reduce(std::forward<F>(func), xs...));
    } else {
        return x;
    }
}

template<class T, class... Ts>
T reduce_min(T x, Ts... xs) {
    return reduce([](auto a, auto b) { return std::min(a, b); }, x, xs...);
}

template<class T, class... Ts>
T reduce_max(T x, Ts... xs) {
    return reduce([](auto a, auto b) { return std::max(a, b); }, x, xs...);
}

static inline bool is_overlap(const cv::Point2f pts1[4], const cv::Point2f pts2[4]) {
    cv::Rect2f box1, box2;
    box1.x = reduce_min(pts1[0].x, pts1[1].x, pts1[2].x, pts1[3].x);
    box1.y = reduce_min(pts1[0].y, pts1[1].y, pts1[2].y, pts1[3].y);
    box1.width = reduce_max(pts1[0].x, pts1[1].x, pts1[2].x, pts1[3].x) - box1.x;
    box1.height = reduce_max(pts1[0].y, pts1[1].y, pts1[2].y, pts1[3].y) - box1.y;
    box2.x = reduce_min(pts2[0].x, pts2[1].x, pts2[2].x, pts2[3].x);
    box2.y = reduce_min(pts2[0].y, pts2[1].y, pts2[2].y, pts2[3].y);
    box2.width = reduce_max(pts2[0].x, pts2[1].x, pts2[2].x, pts2[3].x) - box2.x;
    box2.height = reduce_max(pts2[0].y, pts2[1].y, pts2[2].y, pts2[3].y) - box2.y;
    return (box1 & box2).area() > 0;
}

static inline int argmax(const float* ptr, int len) {
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg])
            max_arg = i;
    }
    return max_arg;
}

constexpr float inv_sigmoid(float x) {
    return -std::log(1 / x - 1);
}

constexpr float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

SmartModel::SmartModel(const Attributes& a, const std::string& onnx_file) {
    this->points_num = a.points_num;
    this->colors_num = a.colors_num;
    this->tags_num = a.tags_num;
    this->keep_thres = a.keep_thres;
    this->RGBcvt = a.RGBcvt;

    net = cv::dnn::readNetFromONNX(onnx_file);

    try {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::Mat input(640, 640, CV_8UC3);
        auto x = cv::dnn::blobFromImage(input) / 255.;
        net.setInput(x);
        net.forward();
        is_openvino = true;
    } catch (cv::Exception&) {
        ::base::print(
            ::base::PrintMode::WARNING,
            "auto_aim.detector",
            "当前无 OpenVINO 环境, 将用 OpenCV 默认方式读取 .onnx."
        );
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        is_openvino = false;
    }
}

std::vector<bbox_t> SmartModel::operator()(const cv::Mat& _img) {
    try {
        cv::Mat img;
        float scale = 640.f / std::max(_img.cols, _img.rows);
        cv::resize(_img, img, { (int)round(_img.cols * scale), (int)round(_img.rows * scale) });
        cv::Mat input(640, 640, CV_8UC3, 127);
        img.copyTo(input({ 0, 0, img.cols, img.rows }));
        if (this->RGBcvt) {
            cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        }

        auto x = cv::dnn::blobFromImage(input) / 255;
        net.setInput(x);
        auto y = net.forward();
        std::vector<bbox_t> before_nms, boxes;
        for (int i = 0; i < y.size[1]; i++) {
            float* result = (float*)y.data + i * y.size[2];
            if (result[8] < inv_sigmoid(KEEP_THRES))
                continue;
            if (result[8] < inv_sigmoid(this->keep_thres[argmax(
                    result + 2 * this->points_num + 1 + this->colors_num,
                    this->tags_num
                )]))
                continue;
            bbox_t box;
            for (int i = 0; i < 4; i++) {
                box.pts[i].x = (result[i * 2 + 0]) / scale;
                box.pts[i].y = (result[i * 2 + 1]) / scale;
            }
            box.color_id = argmax(result + 2 * this->points_num + 1, this->colors_num);
            box.tag_id =
                argmax(result + 2 * this->points_num + 1 + this->colors_num, this->tags_num);
            box.confidence = sigmoid(result[2 * this->points_num]);
            box.confidence_cls =
                sigmoid(result[2 * this->points_num + 1 + this->colors_num + box.tag_id]);
            before_nms.emplace_back(box);
        }
        std::sort(before_nms.begin(), before_nms.end(), [](bbox_t& b1, bbox_t& b2) {
            return b1.confidence > b2.confidence;
        });
        boxes.clear();
        boxes.reserve(before_nms.size());
        std::vector<bool> is_removed(before_nms.size());
        for (int i = 0; i < before_nms.size(); i++) {
            if (is_removed[i])
                continue;
            boxes.push_back(before_nms[i]);
            for (int j = i + 1; j < before_nms.size(); j++) {
                if (is_removed[j])
                    continue;
                if (is_overlap(before_nms[i].pts, before_nms[j].pts))
                    is_removed[j] = true;
            }
        }
        return boxes;
    } catch (std::exception& e) {
        std::ofstream ofs("../log/warning.txt", std::ios::app);
        time_t t;
        time(&t);
        ofs << asctime(localtime(&t)) << "\t" << e.what() << std::endl;
        return std::vector<bbox_t>();
    }
}
