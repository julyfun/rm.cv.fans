#ifndef AIMER_AUTO_AIM_PREDICTOR_PNP_PNP_HPP
#define AIMER_AUTO_AIM_PREDICTOR_PNP_PNP_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "aimer/auto_aim/base/defs.hpp"
#include "aimer/base/armor_defs.hpp"
#include "aimer/base/robot/coord_converter.hpp"

namespace aimer {
aimer::ArmorInfo detected_to_info(
    const aimer::DetectedArmor& detected,
    const aimer::SampleArmor& sample,
    const double& armor_pitch,
    const aimer::CoordConverter* const converter);
// pass
}  // namespace aimer

#endif /* AIMER_AUTO_AIM_PREDICTOR_PNP_PNP_HPP */
