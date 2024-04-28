/*卡尔曼滤波器源代码*/
// 本单元不再维护
#ifndef PREDICTOR_KALMAN_HPP
#define PREDICTOR_KALMAN_HPP

#include <Eigen/Dense>

namespace aimer {
// predict_forward
template<int V_Z, int V_X>  // 例: 1, 2
class Kalman {
  public:
    using Matrix_zzd = Eigen::Matrix<double, V_Z, V_Z>;
    using Matrix_xxd = Eigen::Matrix<double, V_X, V_X>;
    using Matrix_zxd = Eigen::Matrix<double, V_Z, V_X>;
    using Matrix_xzd = Eigen::Matrix<double, V_X, V_Z>;
    using Matrix_x1d = Eigen::Matrix<double, V_X, 1>;
    using Matrix_z1d = Eigen::Matrix<double, V_Z, 1>;

  private:
    Matrix_x1d x_k1;  // k-1时刻的滤波值，即是k-1时刻的值
    Matrix_zxd H;  // 观测矩阵
    Matrix_xxd Q;  // 预测过程噪声偏差的方差
    Matrix_zzd default_R;  // 测量噪声偏差，(系统搭建好以后，通过测量统计实验获得)
    Matrix_xxd P;  // 估计误差协方差
    int predict_order;
    double t = 0.;
    static constexpr double INF = 998244353.;  // 效果好

  public:
    Kalman(const std::vector<double>& Q, const double& R, const int& predict_order) {
        this->H = Matrix_zxd::Zero();
        this->H(0, 0) = 1;
        this->Q = Matrix_xxd::Zero();
        for (int i = 0; i < V_X; ++i)
            this->Q(i, i) = Q[i];
        this->default_R = Matrix_zzd::Zero();
        this->default_R(0, 0) = R;
        this->predict_order = predict_order;
        this->x_k1 = Matrix_x1d::Zero();
        this->t = 0.;
        this->P = Matrix_xxd::Ones() * this->INF;  // Identity 背锅
    }

    Matrix_x1d get_x_k1() const {
        return this->x_k1;
    }

    void init() {
        this->x_k1 = Matrix_x1d::Zero();
        this->P = Matrix_xxd::Ones() * this->INF;
    }

    void set_x(const double& x) {
        this->x_k1(0, 0) = x;
    }

    // 传入绝对时间
    Matrix_x1d predict(const double& t) const {
        Matrix_xxd A = Matrix_xxd::Zero();
        for (int i = 0; i < std::min(this->predict_order, V_X); ++i)
            A(i, i) = 1.;
        for (int i = 1; i < std::min(this->predict_order, V_X); ++i)
            A(i - 1, i) = t - this->t;
        for (int i = 2; i < std::min(this->predict_order, V_X); ++i)
            A(i - 2, i) = 0.5 * (t - this->t) * (t - this->t);
        Matrix_x1d p_x_k = A * this->x_k1;
        return p_x_k;
    }

    void update(const double& x, const double& t) {
        Matrix_z1d z_k(x);
        this->update(z_k, t, this->default_R);
    }

    void update(const double& x, const double& t, const double& R) {
        Matrix_z1d z_k(x);
        Matrix_zzd z_k_R(R);
        this->update(z_k, t, z_k_R);
    }

    void update(const Matrix_z1d& z_k, const double& t, const Matrix_zzd& R) {
        Matrix_xxd A = Matrix_xxd::Zero();
        for (int i = 0; i < V_X; ++i)
            A(i, i) = 1.;
        for (int i = 1; i < V_X; i++)
            A(i - 1, i) = t - this->t;
        for (int i = 2; i < V_X; i++)
            A(i - 2, i) = 0.5 * (t - this->t) * (t - this->t);
        Matrix_x1d p_x_k = A * this->x_k1;
        this->P = A * this->P * A.transpose() + this->Q;
        // 一维时，方差和数值挂钩，因为方差不是比例
        Matrix_xzd K =
            this->P * this->H.transpose() * (this->H * this->P * this->H.transpose() + R).inverse();
        this->x_k1 = p_x_k + K * (z_k - this->H * p_x_k);
        this->P = (Matrix_xxd::Identity() - K * this->H) * this->P;
        this->t = t;
    }
};
}  // namespace aimer
#endif /* _KALMAN_H_ */
