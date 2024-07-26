#include "aimer/base/debug/debug.hpp"

#include "base/param/parameter.hpp"
#include "core_io/robot.hpp"

namespace aimer::debug {

auto draw_line(
    cv::Mat& image,
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const cv::Scalar& color,
    const int& thickness
) -> void {
    cv::line(image, pt1, pt2, color, thickness);
}

auto draw_lines(
    cv::Mat& image,
    const std::vector<cv::Point2f>& pts,
    const cv::Scalar& color,
    const int& thickness,
    const bool& closed
) -> void {
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (!closed && i == pts.size() - 1) {
            break;
        }
        int p = (i + 1) % pts.size();
        cv::line(image, pts[i], pts[p], color, thickness);
    }
}

auto draw_arrow(
    cv::Mat& image,
    const cv::Point2f& pt1,
    const cv::Point2f& pt2,
    const cv::Scalar& color,
    const int& thickness
) -> void {
    cv::line(image, pt1, pt2, color, thickness);
    Eigen::Vector2d vec { pt2.x - pt1.x, pt2.y - pt1.y };
    Eigen::Vector2d vec_l = aimer::math::rotate(vec, M_PI / 4. * 3.) / 7.;
    cv::Point2f pt_l { pt2.x + (float)vec_l(0, 0), pt2.y + (float)vec_l(1, 0) };
    cv::line(image, pt2, pt_l, color, thickness);
    Eigen::Vector2d vec_r = aimer::math::rotate(vec, -M_PI / 4. * 3.) / 7.;
    cv::Point2f pt_r { pt2.x + (float)vec_r(0, 0), pt2.y + (float)vec_r(1, 0) };
    cv::line(image, pt2, pt_r, color, thickness);
}

cv::Scalar heightened_color(const cv::Scalar& color, const double& z) {
    cv::Scalar res;
    for (int i = 0; i < 3; ++i) {
        res[i] = z >= 0. ? 255. - (255. - color[i]) * std::pow(0.5, z / FLASK_MAP_PETER_BY_BRIGHT)
                         : color[i] * std::pow(0.5, -z / FLASK_MAP_PETER_BY_BRIGHT);
    }
    return res;
}

FlaskPoint pos_to_map_point(
    const Eigen::Vector3d& pos,
    const cv::Scalar& color,
    const int& radius,
    const int& thickness
) {
    return FlaskPoint(
        { float(
              FLASK_MAP_MID_X
              + pos(0, 0) * base::get_param<double>("auto-aim.debug.flask.map.pixel-per-meter")
          ),
          float(
              FLASK_MAP_MID_Y
              - pos(1, 0) * base::get_param<double>("auto-aim.debug.flask.map.pixel-per-meter")
          ) },
        heightened_color(color, pos(2, 0)),
        radius,
        thickness
    );
}

std::vector<FlaskLine> pts_to_map_lines(
    const std::vector<cv::Point2f>& pts,
    const cv::Scalar& color,
    const bool& closed,
    const int& thickness
) {
    // 一条一条放入
    std::vector<FlaskLine> lines;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        int p = (i + 1) % pts.size();
        if (p == 0 && !closed) {
            break;
        }
        lines.emplace_back(std::pair<cv::Point2f, cv::Point2f>(pts[i], pts[p]), color, thickness);
    }
    return lines;
}

std::vector<FlaskLine> poses_to_map_lines(
    const std::vector<Eigen::Vector3d>& poses,
    const cv::Scalar& color,
    const bool& closed,
    const int& thickness
) {
    // 一条一条放入
    std::vector<FlaskLine> lines;
    for (std::size_t i = 0; i < poses.size(); ++i) {
        int p = (i + 1) % poses.size();
        if (p == 0 && !closed) {
            break;
        }
        FlaskPoint pt1 = pos_to_map_point(poses[i], color, 0, 0);
        FlaskPoint pt2 = pos_to_map_point(poses[p], color, 0, 0);
        lines.emplace_back(
            std::pair<cv::Point2f, cv::Point2f>(pt1.pt, pt2.pt),
            (pt1.color + pt2.color) / 2.,
            thickness
        );
    }
    return lines;
}

std::vector<FlaskLine> pos_pair_to_map_arrow(
    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& pos_pair,
    const cv::Scalar& color,
    const int& thickness
) {
    // 一条一条放入
    std::vector<FlaskLine> lines;
    FlaskPoint pt1 = pos_to_map_point(pos_pair.first, color, 0, 0);
    FlaskPoint pt2 = pos_to_map_point(pos_pair.second, color, 0, 0);
    lines.emplace_back(std::make_pair(pt1.pt, pt2.pt), (pt1.color + pt2.color) / 2., thickness);
    Eigen::Vector2d vec { pt2.pt.x - pt1.pt.x, pt2.pt.y - pt1.pt.y };
    Eigen::Vector2d vec_l = aimer::math::rotate(vec, M_PI / 6. * 5.) / 7.;
    cv::Point2f pt_l { pt2.pt.x + (float)vec_l(0, 0), pt2.pt.y + (float)vec_l(1, 0) };
    lines.emplace_back(std::make_pair(pt2.pt, pt_l), pt2.color, thickness);
    Eigen::Vector2d vec_r = aimer::math::rotate(vec, -M_PI / 6. * 5.) / 7.;
    cv::Point2f pt_r { pt2.pt.x + (float)vec_r(0, 0), pt2.pt.y + (float)vec_r(1, 0) };
    lines.emplace_back(std::make_pair(pt2.pt, pt_r), pt2.color, thickness);
    return lines;
}

FlaskText pos_str_to_map_text(
    const std::string& str,
    const Eigen::Vector3d& pos,
    const cv::Scalar& color,
    const double& scale
) {
    return FlaskText(str, pos_to_map_point(pos, { 0., 0., 0. }, 0, .0).pt, color, scale);
}

FlaskStream flask_aim;
FlaskStream flask_map;

auto Stm32Shoot::add(const int& id, const double& img_t) -> void {
    // 时间超过 t + latency 后可以发射
    if (this->pending_signals.size() + 1 <= Stm32Shoot::MAX_SZ) {
        this->pending_signals.push_back(Stm32Shoot::IdT { id, img_t });
    }
}

auto Stm32Shoot::get_last_shoot_id(const double& img_t) -> int {
    // 实际上是传输过去有延迟，
    while (!this->pending_signals.empty()
           && img_t >= this->pending_signals.front().img_t + Stm32Shoot::SHOOT_LATENCY)
    {
        // 信号已经到达，进行信号处理
        if (this->pending_signals.front().img_t >= this->last_shoot.img_t
                + base::get_param<double>("auto-aim.ec-simulator.shoot-interval"))
        {
            this->last_shoot = this->pending_signals.front();
        }
        this->pending_signals.pop_front();
    }
    return this->last_shoot.id;
}

/** @class ProcessTimer */
void ProcessTimer::process_begin() {
    this->begin_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    );
}

void ProcessTimer::print_process_time(const char* str) const {
    std::chrono::microseconds cur = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    );
    std::cout << str << ": " << cur.count() - this->begin_time.count() << " micros.\n";
}

double ProcessTimer::get_process_time() const {
    std::chrono::microseconds cur = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    );
    return (cur.count() - this->begin_time.count()) / 1e6;
}

ProcessTimer process_timer;
RegisterTimer register_timer;

} // namespace aimer::debug
