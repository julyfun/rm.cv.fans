<p align="center">
  <a href="https://sjtu-robomaster-team.github.io/"><img src="https://sjtu-robomaster-team.github.io/assets/img/logo.png" alt="Logo" height=140></a>
</p>
<h1 align="center">Lmtd - Linear Modelled Top Detector</h1>

<div align="center">
  <a href="https://sjtu-robomaster-team.github.io/antitop/">汇报</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://www.bilibili.com/video/BV1vX4y1W7U7">讲解</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/julyfun/rm.cv.fans/issues/new">Issues</a>
  <br />
</div>

## What is Lmtd?

手撸的通信，参数，UI，数学和坐标变换，兼容性一般但是非常快。

- 运动方程和精细火控 [aimer/auto_aim/predictor/motion/lmtd_top_model.cpp](aimer/auto_aim/predictor/motion/lmtd_top_model.cpp)
- 降低自由度获取比 pnp 精准得多的姿态 [aimer/auto_aim/predictor/motion/top_model.cpp](aimer/auto_aim/predictor/motion/top_model.cpp)
- 弹道校正 [aimer/auto_aim/predictor/aim/aim_corrector.cpp](aimer/auto_aim/predictor/aim/aim_corrector.cpp)
- 运行时热更新参数 [base/param/parameter.cpp](base/param/parameter.cpp)
- 通用 EKF 滤波器 [aimer/base/math/filter/adaptive_ekf.hpp](aimer/base/math/filter/adaptive_ekf.hpp)
- 手搓坐标变换和精准弹道估计 [aimer/base/robot/coord_converter.cpp](aimer/base/robot/coord_converter.cpp)

https://github.com/julyfun/assets/assets/43675484/2e8579c5-c233-410b-8f3e-c151fb9df543

https://github.com/julyfun/rm.cv.fans/assets/43675484/6342e070-9881-457e-a122-8b27a3508e6d

