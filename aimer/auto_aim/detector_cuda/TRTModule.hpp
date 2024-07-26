//
// Created by xinyang on 2021/4/8.
//
// Edited by shenyibo on 2022/7/4.
//

#ifndef _ONNXTRTMODULE_HPP_
#define _ONNXTRTMODULE_HPP_

#include <NvInfer.h>

#include <opencv2/core.hpp>

struct alignas(4) bbox_t {
    cv::Point2f pts[4];  // [pt0, pt1, pt2, pt3]
    float confidence;
    float confidence_cls;
    int color_id;  // 0: blue, 1: red, 2: gray, 3: purple
    int tag_id;  // 0: guard, 1-5: number, 6: outpost, 7 8: base
    bool operator<(bbox_t b) const {
        int rank[10] = {3, 4, 5, 6, 7, 8, 0, 1, 2, 9};

        if (rank[tag_id] > rank[b.tag_id]) {
            return false;
        } else if (rank[tag_id] < rank[b.tag_id]) {
            return true;
        } else if (confidence * confidence_cls < b.confidence * b.confidence_cls) {
            return false;
        } else {
            return true;
        }
    }
};

struct Attributes {
    int points_num;
    int colors_num;
    int tags_num;
    float* keep_thres;
    bool RGBcvt;
};

/*
 * 四点模型
 */
class TRTModule {
    static constexpr int TOPK_NUM = 128;
    static constexpr float KEEP_THRES = 0.3f;

    int points_num;
    int colors_num;
    int tags_num;
    float* keep_thres;
    bool RGBcvt;

  public:
    explicit TRTModule(const Attributes& a, const std::string& onnx_file);

    ~TRTModule();

    TRTModule(const TRTModule&) = delete;

    TRTModule operator=(const TRTModule&) = delete;

    std::vector<bbox_t> operator()(const cv::Mat& src) const;

    std::vector<bbox_t> operator()(const cv::Mat& src, cv::Rect& rect) const;

    Attributes getAttributes();

  private:
    void build_engine_from_onnx(const std::string& onnx_file);

    void build_engine_from_cache(const std::string& cache_file);

    void cache_engine(const std::string& cache_file);

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    mutable void* device_buffer[2];
    float* output_buffer;
    cudaStream_t stream;
    int input_idx, output_idx;
    size_t input_sz, output_sz;
};

#endif /* _ONNXTRTMODULE_HPP_ */
