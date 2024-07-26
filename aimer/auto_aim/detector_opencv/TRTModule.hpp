//
// Created by xinyang on 2021/4/28.
//
// Edited by shenyibo on 2022/6/13. its for opencv

#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include <opencv2/opencv.hpp>
#include <string>

struct alignas(4) bbox_t {
    cv::Point2f pts[5];  // [pt0, pt1, pt2, pt3]
    float confidence;
    float confidence_cls;
    int color_id;  // 0: blue, 1: red, 2: gray, 3: purple
    int tag_id;  // 0: guard, 1-5: number, 6: outpost, 7 8: base
};

struct Attributes {
    int points_num;
    int colors_num;
    int tags_num;
    float* keep_thres;
    bool RGBcvt;
};

class SmartModel {
    int points_num;
    int colors_num;
    int tags_num;
    float* keep_thres;
    bool RGBcvt;

  public:
    explicit SmartModel(const Attributes& a, const std::string& onnx_file);

    SmartModel(const SmartModel& module) = delete;

    std::vector<bbox_t> operator()(const cv::Mat& img);

    bool with_openvino() const {
        return is_openvino;
    }

    Attributes getAttributes();

  private:
    static constexpr int TOPK_NUM = 128;
    static constexpr float KEEP_THRES = 0.1f;

    cv::dnn::Net net;
    bool is_openvino = false;
};

#endif /* _MODEL_HPP_ */
