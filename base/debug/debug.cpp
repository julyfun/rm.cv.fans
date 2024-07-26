#include "base/debug/debug.hpp"

#include <fmt/color.h>
#include <fmt/format.h>

namespace base {
const std::unordered_map<base::PrintMode, fmt::color> PRINT_COLOR = {
    { PrintMode::INFO, fmt::color::cornflower_blue },
    { PrintMode::WARNING, fmt::color::orange },
    { PrintMode::ERROR, fmt::color::red },
    { PrintMode::PANIC, fmt::color::red }
};

const std::unordered_map<base::PrintMode, std::string> PRINT_PREFIX = {
    { PrintMode::INFO, "[INFO]" },
    { PrintMode::WARNING, "[WARNING]" },
    { PrintMode::ERROR, "[ERROR]" },
    { PrintMode::PANIC, "[PANIC]" }
};

} // namespace base
