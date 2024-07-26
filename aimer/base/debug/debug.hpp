#ifndef AIMER_BASE_DEBUG_DEBUG_HPP
#define AIMER_BASE_DEBUG_DEBUG_HPP

#include <fmt/format.h>

#include <Eigen/Dense>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "aimer/base/armor_defs.hpp"
#include "base/param/parameter.hpp"
#include "common/common.hpp"
#include "core_io/robot.hpp"

namespace aimer::debug {

// typename 签名类似于 #define
// 通过自定义特征函数指针记录一个时期内的特征值（比如最大值）
template<typename ValueT, typename PeriodT>
class PeriodicRecorder {
public:
    PeriodicRecorder(
        const std::function<bool(const ValueT&, const ValueT&)>& cmp,
        const PeriodT& init_stamp
    ):
        cmp(cmp),
        stamp(init_stamp) {}
    void update(const ValueT& value, const PeriodT& stamp, const PeriodT& period) {
        if (stamp - this->stamp >= period) {
            // 本帧超时则本帧不计入
            this->last_record = this->record;
            this->record = value;
            this->stamp = stamp;
        } else {
            if (this->cmp(this->record, value)) {
                this->record = value;
            }
        }
    }
    ValueT get() const {
        return this->last_record;
    }

private:
    std::function<bool(const ValueT&, const ValueT&)> cmp;
    ValueT last_record = ValueT();
    ValueT record = ValueT();
    PeriodT stamp;
};

template<typename ValueT, typename PeriodT>
class PeriodicAverage {
public:
    PeriodicAverage(const std::size_t& buf_size, const PeriodT& init_stamp):
        buf_size(buf_size),
        stamp(init_stamp) {}

    void update(const ValueT& value, const PeriodT& stamp, const PeriodT& period) {
        if (stamp - this->stamp >= period) {
            this->last_period_ave = this->data.empty() ? ValueT() : ([&]() {
                std::size_t size = this->data.size();
                ValueT tot = this->data.front();
                this->data.pop();
                while (!this->data.empty()) {
                    tot += this->data.front();
                    this->data.pop();
                }
                return tot * (1. / double(size));
            }());
            this->stamp = stamp;
        }
        if (this->data.size() + 1u > this->buf_size && !this->data.empty()) {
            this->data.pop();
        }
        if (this->data.size() + 1u <= this->buf_size) {
            this->data.push(value);
        }
    }

    ValueT get() const {
        return this->last_period_ave;
    }

private:
    const std::size_t buf_size;
    ValueT last_period_ave = ValueT();
    std::queue<ValueT> data;
    PeriodT stamp;
};

const int FLASK_MAP_WIDTH = 1000;
const int FLASK_MAP_HEIGHT = 1000;
const int FLASK_MAP_MID_X = FLASK_MAP_WIDTH / 2;
const int FLASK_MAP_MID_Y = FLASK_MAP_HEIGHT / 2;
const double FLASK_MAP_PETER_BY_BRIGHT = 1.;
const cv::Scalar RED_MAIN_COLOR = { 63., 127., 255. };
const cv::Scalar BLUE_MAIN_COLOR = { 255., 127., 63. };
const cv::Scalar GRAY_MAIN_COLOR = { 190., 190., 190. };
const cv::Scalar MOTION_MAIN_COLOR = { 253, 238, 173 };
const cv::Scalar FLASK_MAP_TOP_CENTER_COLOR = { 255., 0., 255. };
const cv::Scalar FLASK_MAP_TOP_EDGE_COLOR = { 150., 135., 189. };
const cv::Scalar FLASK_MAP_TOP_PT_COLOR = { 0., 255., 0. };
const cv::Scalar FLASK_MAP_TEXT_COLOR = { 127, 127, 127 };
const double FLASK_MAP_TEXT_SCALE = 1.;
const int FLASK_MAP_PT_RADIUS = 4;
const int FLASK_MAP_THICKNESS = 2;

// bool in_flask(const cv::Point2f& pt) {
//   return 0.f < pt.x && pt.x < float(FLASK_AIM_WIDTH) && 0.f < pt.y &&
//          pt.y < float(FLASK_ORI_HEIGHT);
// }

auto draw_line(
    cv::Mat& image,
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const cv::Scalar& color,
    const int& thickness
) -> void;

auto draw_lines(
    cv::Mat& image,
    const std::vector<cv::Point2f>& pts,
    const cv::Scalar& color,
    const int& thickness,
    const bool& closed
) -> void;

auto draw_arrow(
    cv::Mat& image,
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const cv::Scalar& color,
    const int& thickness
) -> void;

struct FlaskPoint {
    FlaskPoint(
        const cv::Point2f& pt,
        const cv::Scalar& color,
        const int& radius,
        const int& thickness
    ):
        pt(pt),
        color(color),
        radius(radius),
        thickness(thickness) {}
    cv::Point2f pt;
    cv::Scalar color;
    int radius;
    int thickness;
};

struct FlaskLine {
    FlaskLine(
        const std::pair<cv::Point2f, cv::Point2f>& pt_pair,
        const cv::Scalar& color,
        const int& thickness
    ):
        pt_pair(pt_pair),
        color(color),
        thickness(thickness) {}
    std::pair<cv::Point2f, cv::Point2f> pt_pair;
    cv::Scalar color;
    int thickness;
};

struct FlaskText {
    FlaskText(
        const std::string& str,
        const cv::Point2f& pt,
        const cv::Scalar& color,
        const double& scale
    ):
        str(str),
        pt(pt),
        color(color),
        scale(scale) {}
    std::string str;
    cv::Point2f pt;
    cv::Scalar color;
    double scale;
};

cv::Scalar heightened_color(const cv::Scalar& color, const double& z);

FlaskPoint pos_to_map_point(
    const Eigen::Vector3d& pos,
    const cv::Scalar& color,
    const int& radius,
    const int& thickness
);

std::vector<FlaskLine> pts_to_map_lines(
    const std::vector<cv::Point2f>& pts,
    const cv::Scalar& color,
    const bool& closed,
    const int& thickness
);

std::vector<FlaskLine> poses_to_map_lines(
    const std::vector<Eigen::Vector3d>& poses,
    const cv::Scalar& color,
    const bool& closed,
    const int& thickness
);

std::vector<FlaskLine> pos_pair_to_map_arrow(
    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pos_pair,
    const cv::Scalar& color,
    const int& thickness
);

FlaskText pos_str_to_map_text(
    const std::string& str,
    const Eigen::Vector3d& pos,
    const cv::Scalar& color,
    const double& scale
);

class FlaskStream {
public:
    // 静态对象，唯一指针，应该避免多重定义
    FlaskStream& operator<<(const char* str) {
        this->logs.emplace_back(str);
        return *this;
    }

    FlaskStream& operator<<(const std::string& str) {
        this->logs.push_back(str);
        return *this;
    }

    FlaskStream& operator<<(const FlaskPoint& pt) {
        this->pts.push_back(pt);
        return *this;
    }

    FlaskStream& operator<<(const FlaskLine& line) {
        this->lines.push_back(line);
        return *this;
    }

    FlaskStream& operator<<(const std::vector<FlaskLine>& lines) {
        for (const auto& line: lines) {
            this->lines.push_back(line);
        }
        return *this;
    }

    FlaskStream& operator<<(const FlaskText& text) {
        this->texts.push_back(text);
        return *this;
    }

    FlaskStream& operator>>(cv::Mat& img) {
        int cnt = 0;
        for (auto& str: this->logs) {
            cv::putText(
                img,
                str,
                { 20, 80 + cnt * 24 },
                cv::FONT_HERSHEY_DUPLEX,
                0.8,
                { 0, 0, 255 }
            );
            ++cnt;
        }
        for (auto& pt: this->pts) {
            cv::circle(img, pt.pt, pt.radius, pt.color, pt.thickness);
        }
        for (auto& line: this->lines) {
            cv::line(img, line.pt_pair.first, line.pt_pair.second, line.color, line.thickness);
        }
        for (auto& text: this->texts) {
            cv::putText(
                img,
                text.str,
                { int(text.pt.x), int(text.pt.y) },
                cv::FONT_HERSHEY_DUPLEX,
                text.scale,
                text.color
            );
        }
        return *this;
    }

    void clear() {
        this->logs.clear();
        this->pts.clear();
        this->lines.clear();
        this->texts.clear();
    }

private:
    std::vector<std::string> logs;
    std::vector<FlaskPoint> pts;
    std::vector<FlaskLine> lines;
    std::vector<FlaskText> texts;
};

/** @brief 存储需要绘制在图像中的图形 */
extern FlaskStream flask_aim;
extern FlaskStream flask_map;

/// @class BulletId 注意是 debug
// 想要实现的功能：模拟子弹的序号
// 几乎每一帧都会有信号
// 电控每一帧能更新的就是最近一发发出子弹的 id
// 我在镜头静止时，只能模拟的是每一发子弹都发射出去
// 视频帧率为二十几帧，不允许每发都发射
// 你给我所有 aim 的 id 和 t，访问时我直接给你最近一个子弹
class Stm32Shoot {
public:
    auto add(const int& id, const double& img_t) -> void;
    auto get_last_shoot_id(const double& img_t) -> int;

private:
    struct IdT {
        int id;
        double img_t;
    };
    std::deque<IdT> pending_signals;
    IdT last_shoot { 0, 0. };
    static constexpr std::size_t MAX_SZ = 100u;
    static constexpr double SHOOT_LATENCY = 60e-3;
};

class ProcessTimer {
public:
    void process_begin();
    void print_process_time(const char* str) const;
    double get_process_time() const;

private:
    std::chrono::microseconds begin_time;
};

extern ProcessTimer process_timer;

struct StartEndTime {
    std::string start;
    std::string end;
    std::chrono::microseconds::rep time;
};

class RegisterTimer {
public:
    auto get_and_register(const std::string& stamp_name) -> debug::StartEndTime {
        debug::StartEndTime res;
        std::chrono::microseconds cur = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        );
        res.start = this->cur_name;
        res.end = stamp_name;
        res.time = cur.count() - this->cur_time.count();
        this->cur_time = cur;
        this->cur_name = stamp_name;
        return res;
    }

private:
    std::string cur_name;
    std::chrono::microseconds cur_time;
};

extern RegisterTimer register_timer;

inline auto start_end_time_to_fmt_pair(const debug::StartEndTime& x)
    -> std::pair<std::string, std::string> {
    return { fmt::format("{}_到_{}_耗时", x.start, x.end), fmt::format("{} 微秒", x.time) };
}

inline auto auto_aim_page() -> std::shared_ptr<base::webview_info::Page> {
    return umt::ObjManager<::base::webview_info::Page>::find("auto_aim.predictor.aim");
}

} // namespace aimer::debug

#endif /* AIMER_BASE_DEBUG_DEBUG_HPP */
