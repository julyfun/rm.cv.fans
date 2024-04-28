/*滤波器模块*/
#ifndef AIMER_BASE_MATH_FILTER_FILTER_HPP
#define AIMER_BASE_MATH_FILTER_FILTER_HPP

#include "aimer/base/math/filter/adaptive_ekf.hpp"
#include "aimer/base/math/filter/kalman.hpp"
#include "aimer/base/math/math.hpp"

// 本文件是为了通过时间获取装甲板的时空坐标、空间速度，而不只是滤波。比较 basic
namespace aimer {
// 对于 N = 1， q_vec {0.}
/**
 * @brief 一维实数观测值的滤波器
 * @param N 预测阶数
 */
template<int N>
class SinglePredict {
public:
    explicit SinglePredict(const double& delta_t): delta_t(delta_t) {}
    template<typename T>
    void operator()(const T x_pre[N], T x_cur[N]) const {
        double coeff[N] = { 1. };
        for (int i = 1; i < N; ++i) {
            coeff[i] = coeff[i - 1] / double(i);
        }
        for (int i = 0; i < N; ++i) {
            x_cur[i] = x_pre[i];
            for (int j = i + 1; j < N; ++j) {
                x_cur[i] += coeff[j - i] * std::pow(this->delta_t, j - i) * x_pre[j];
            }
        }
    }

private:
    double delta_t = 0.;
};

template<int N>
class SingleMeasure {
public:
    template<typename T>
    void operator()(const T x[N], T y[1]) const {
        y[0] = x[0];
    }
};

template<int ORDER>
class SingleFilter {
public:
    using Ekf = aimer::AdaptiveEkf<ORDER, 1>;

    SingleFilter() = default;

    // 这一初始化将会使预测方差为单位矩阵
    // 如果不做这一步，则初值为 0 且内部方差为正无穷，也就是第一个更新值作为标准
    void init_x(const Eigen::Matrix<double, ORDER, 1>& x) {
        this->ekf.init_x(x);
    }

    void set_x(const Eigen::Matrix<double, ORDER, 1>& x) {
        this->ekf.set_x(x);
    }

    void set_t(const double& t) {
        this->ekf_t = t;
    }

    Eigen::Matrix<double, ORDER, 1> get_x() const {
        return this->ekf.get_x();
    }

    Eigen::Matrix<double, ORDER, 1> predict(const double& t) const {
        aimer::SinglePredict<ORDER> predict_func { t - this->ekf_t };
        typename Ekf::PredictResult pre_res = this->ekf.predict(predict_func);
        return pre_res.x_p;
    }

    void update(
        const double& x,
        const double& t,
        const std::vector<double>& q_vec,
        const std::vector<double>& r_vec
    ) {
        aimer::SingleMeasure<ORDER> measure_func;
        aimer::SinglePredict<ORDER> predict_func { t - this->ekf_t };
        this->ekf.update(
            measure_func,
            predict_func,
            aimer::math::vec_x_to_mat_x1<1>({ x }),
            aimer::math::vec_x_to_mat_xx<ORDER>(q_vec),
            aimer::math::vec_x_to_mat_xx<1>(r_vec)
        );
        this->ekf_t = t;
    }

private:
    Ekf ekf;
    double ekf_t = 0.;
};

/**
 * @interface
 */
class PositionPredictorInterface {
public:
    virtual ~PositionPredictorInterface() = default;
    virtual Eigen::Vector3d predict_pos(const double& t) const = 0;
    virtual Eigen::Vector3d predict_v(const double& t) const = 0;

private:
};

class EkfPredict {
public:
    explicit EkfPredict(const double& delta_t): delta_t(delta_t) {}
    template<typename T>
    void operator()(const T x_pre[6], T x_cur[6]) const {
        x_cur[0] = x_pre[0] + this->delta_t * x_pre[1];
        x_cur[1] = x_pre[1];
        x_cur[2] = x_pre[2] + this->delta_t * x_pre[3];
        x_cur[3] = x_pre[3];
        x_cur[4] = x_pre[4] + this->delta_t * x_pre[5];
        x_cur[5] = x_pre[5];
    }

private:
    double delta_t = 0.;
};

class EkfMeasure {
public:
    template<typename T>
    void operator()(const T x[6], T y[3]) const {
        T x0[3] { x[0], x[2], x[4] };
        aimer::math::ceres_xyz_to_ypd(x0, y);
    }
};

// very old template
template<int V_X, int V_Z>
class PositionFilter: public PositionPredictorInterface {
public:
    PositionFilter() = default;

    void set_state(const std::array<std::tuple<Eigen::Matrix<double, V_X, 1>, double>, 3>& state) {
        for (int i = 0; i < 3; i++) {
            this->kalman[i].set_x(std::get<0>(state[i]));
            this->kalman[i].set_t(std::get<1>(state[i]));
        }
    }

    Eigen::Vector3d predict_pos(const double& t) const override { // const 这个是不得不加
        Eigen::Vector3d res;
        res << this->kalman[0].predict(t)(0, 0), this->kalman[1].predict(t)(0, 0),
            this->kalman[2].predict(t)(0, 0);
        return res;
    }

    void set_pos(const Eigen::Vector3d& pos) {
        for (int i = 0; i < 3; i++) {
            auto x = this->kalman[i].get_x();
            x[0] = pos[i];
            this->kalman[i].set_x(x);
        }
    }

    // v-陀螺仪 xyz
    Eigen::Vector3d predict_v(const double& t) const override {
        // 1， 2 means 匀速预测
        // 1， 3 means 匀加速预测
        // V_X = 1 : 转移为 x = x
        // V_X = 2 : 转移为 x = x + v * dt
        // V_X = 3 : 转移为 x = x + v * dt + 0.5 * a * dt * dt
        // predict_order = 1 : 预测为 x = x, v = 0
        if (V_X < 2) {
            return Eigen::Vector3d(0., 0., 0.);
        }
        Eigen::Vector3d res;
        res << this->kalman[0].predict(t)(1, 0), this->kalman[1].predict(t)(1, 0),
            this->kalman[2].predict(t)(1, 0);
        return res;
    }

    void set_v(const Eigen::Vector3d& v) {
        // v[0, 1, 2]
        if (V_X < 2) {
            return;
        }
        for (int i = 0; i < 3; i++) {
            auto x = this->kalman[i].get_x();
            x[1] = v[i]; // x = [1.2m, 0.01m/s]
            this->kalman[i].set_x(x);
        }
    }
    // aimer::math::YpdCoord get_ypd_v() const {
    //   return aimer::math::get_ypd_v(this->get_pos(), this->get_v());
    // }  // 此算法尚不具有不同阶通用性，所有类似的零阶和一阶
    // // 都可修改为阶通用版本

    // // ypd_v_i - 陀螺仪 xyz
    // aimer::math::YpdCoord predict_ypd_v(const double& t) const {
    //   return aimer::math::get_ypd_v(this->predict_pos(t), this->predict_v(t));
    // }  // 此算法尚不具有不同阶通用性，所有类似的零阶和一阶

    void update(
        const Eigen::Vector3d& pos,
        const double& t,
        const std::vector<double>& q_vec,
        const std::vector<double>& r_vec
    ) {
        for (int i = 0; i < 3; ++i) {
            this->kalman[i].update(pos(i, 0), t, q_vec, r_vec);
        }
    }

private:
    aimer::SingleFilter<2> kalman[3];
    std::vector<double> q_vec;
    std::vector<double> r_vec;
};

// 匀速预测
class PositionEkf: public PositionPredictorInterface {
public:
    using Ekf = AdaptiveEkf<6, 3>;

    PositionEkf() = default;

    Eigen::Vector3d predict_pos(const double& t) const override {
        aimer::EkfPredict predict_func { t - this->ekf_t };
        Ekf::PredictResult pre_res = this->ekf.predict(predict_func);
        return Eigen::Vector3d(pre_res.x_p(0, 0), pre_res.x_p(2, 0), pre_res.x_p(4, 0));
    }

    Eigen::Vector3d predict_v(const double& t) const override {
        aimer::EkfPredict predict_func { t - this->ekf_t };
        Ekf::PredictResult pre_res = this->ekf.predict(predict_func);
        return Eigen::Vector3d(pre_res.x_p(1, 0), pre_res.x_p(3, 0), pre_res.x_p(5, 0));
    }

    void init(const aimer::math::YpdCoord& ypd, const double& t) {
        Eigen::Vector3d pos = aimer::math::ypd_to_xyz(ypd);
        std::vector<double> pos_vec { pos(0, 0), 0., pos(1, 0), 0., pos(2, 0), 0. };
        this->ekf.init_x(aimer::math::vec_x_to_mat_x1<6>(pos_vec));
        this->ekf_t = t;
    }

    // aimer::math::YpdCoord predict_ypd_v(const double& t) const {
    //   return aimer::math::get_ypd_v(this->predict_pos(t), this->predict_v(t));
    // }

    void update(
        const aimer::math::YpdCoord& ypd,
        const double& t,
        const std::vector<double>& q_vec,
        const std::vector<double>& r_vec
    ) {
        aimer::EkfMeasure measure_func;
        aimer::EkfPredict predict_func { t - this->ekf_t };
        this->ekf.predict_forward(predict_func, aimer::math::vec_x_to_mat_xx<6>(q_vec));
        Ekf::MeasureResult mea_res = this->ekf.measure(measure_func);
        aimer::math::YpdCoord ypd_p { mea_res.y_e(0, 0), mea_res.y_e(1, 0), mea_res.y_e(2, 0) };
        // predict_forward 之后的内部 ypd 需和数据 ypd 做一次 get_closest，以防
        // (-179) - (+179) 的差角为 358 度。ceres:Jet 是一阶展开，无法处理 358 度
        std::vector<double> ypd_close { aimer::math::get_closest(ypd.yaw, ypd_p.yaw, 2. * M_PI),
                                        aimer::math::get_closest(ypd.pitch, ypd_p.pitch, 2. * M_PI),
                                        ypd.dis };
        this->ekf.update_forward(
            measure_func,
            aimer::math::vec_x_to_mat_x1<3>(ypd_close),
            aimer::math::vec_x_to_mat_xx<3>(r_vec)
        );
        this->ekf_t = t;
    }

private:
    Ekf ekf;
    double ekf_t = 0.;
};

template<int V_Z, int V_X>
class AngleFilter {
public:
    // 禁用 predict_order
    AngleFilter(const double& range, const std::vector<double>& Q, const double& R):
        filter(Kalman<V_Z, V_X>(Q, R, V_X)),
        range(range) {}

    void update(const double& angle, const double& t) {
        this->filter.update(
            aimer::math::get_closest(angle, this->filter.predict(t)(0, 0), this->range),
            t
        );
        // 强制修正内部更新的角度
        this->filter.set_x(aimer::math::reduced(this->filter.get_x_k1()(0, 0), this->range));
    }

    void update(const double& angle, const double& t, const double& R) {
        this->filter.update(
            aimer::math::get_closest(angle, this->filter.predict(t)(0, 0), this->range),
            t,
            R
        );
        this->filter.set_x(aimer::math::reduced(this->filter.get_x_k1()(0, 0), this->range));
    }

    void init() {
        this->filter.init();
    }

    double get_angle() const {
        return this->filter.get_x_k1()(0, 0);
    }

    double get_w() const {
        return this->filter.get_x_k1()(1, 0);
    }

    double predict_angle(const double& t) const {
        return aimer::math::reduced(this->filter.predict(t)(0, 0), this->range);
    }

private:
    Kalman<V_Z, V_X> filter;
    double range;
};
} // namespace aimer

#endif /* AIMER_BASE_MATH_FILTER_FILTER_HPP */
