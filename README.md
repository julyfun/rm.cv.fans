<p align="center">
  <a href="https://sjtu-robomaster-team.github.io/"><img src="https://sjtu-robomaster-team.github.io/assets/img/logo.png" alt="Logo" height=140></a>
</p>
<h1 align="center">LMTD - Linear Modelled Top Detector</h1>

<div align="center">
  <a href="https://sjtu-robomaster-team.github.io/antitop/">汇报</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://www.bilibili.com/video/BV1vX4y1W7U7">视频</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/julyfun/rm.cv.fans/issues/new">Issues</a>
  <br />
</div>

## What is LMTD?

手搓通信，参数，UI，数学和坐标变换，兼容性一般，但是非常快。

- LMTD 运动方程，"Water-gun" 火力控制，每一段延迟都精细考虑 [lmtd_top_model.cpp](aimer/auto_aim/predictor/motion/lmtd_top_model.cpp)
- 降低自由度计算装甲板 Rotation，相比 PnP 精度显著提升 [top_model.cpp](aimer/auto_aim/predictor/motion/top_model.cpp)
- 精打细算的延迟匹配文档 [latency.md](docs/auto_aim/latency.md)
- 平移运动模型，收敛快速 [armor_model.cpp](aimer/auto_aim/predictor/motion/armor_model.cpp)
- Runtime 热更新的参数表 [parameter.cpp](base/param/parameter.cpp)
- 精准弹道解算 [coord_converter.cpp](aimer/base/robot/coord_converter.cpp)
- 自主弹道校正 [aim_corrector.cpp](aimer/auto_aim/predictor/aim/aim_corrector.cpp)
- 通用 EKF 类 [adaptive_ekf.hpp](aimer/base/math/filter/adaptive_ekf.hpp)

https://github.com/julyfun/assets/assets/43675484/2e8579c5-c233-410b-8f3e-c151fb9df543

https://github.com/julyfun/rm.cv.fans/assets/43675484/6342e070-9881-457e-a122-8b27a3508e6d

