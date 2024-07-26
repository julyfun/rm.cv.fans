#ifndef BASE_DEBUG_DEBUG_HPP
#define BASE_DEBUG_DEBUG_HPP

// ::fmt

#include <unordered_map>

#include <Eigen/Core>
#include <fmt/color.h>

#include "UltraMultiThread/include/umt/umt.hpp"
#include "common/common.hpp"

namespace base {
// use ::fmt;
namespace fmt = ::fmt;

const Eigen::IOFormat EIGEN_HEAVY_FORMAT = {
    Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]"
};

template<typename T>
std::string streamed_to_str(const T& x) {
    std::stringstream ss;
    ss << x;
    return ss.str();
}

template<typename T>
std::string mat_to_str(const T& mat) {
    return streamed_to_str(mat.format(EIGEN_HEAVY_FORMAT));
}

enum class PrintMode { INFO, WARNING, ERROR, PANIC };

extern const std::unordered_map<base::PrintMode, fmt::color> PRINT_COLOR;

extern const std::unordered_map<base::PrintMode, std::string> PRINT_PREFIX;

template<typename... T>
void print(
    const base::PrintMode& mode,
    const std::string& node_name,
    const std::string& content,
    T&&... args
) {
    fmt::print(
        fmt::fg(base::PRINT_COLOR.at(mode)),
        base::PRINT_PREFIX.at(mode) + " " + (node_name == "" ? "" : "@" + node_name + ": ")
            + content + '\n',
        args...
    );
}

template<typename... T>
void print_info(const std::string& node, const std::string& content, T&&... args) {
    print(PrintMode::INFO, node, content, args...);
}

template<typename... T>
[[deprecated]] void println(const std::string& content, T&&... args) {
    print(PrintMode::INFO, "", content, args...);
}

inline std::shared_ptr<::base::webview_info::Page> webview_info_hold(const std::string& page) {
    return ::umt::ObjManager<::base::webview_info::Page>::find_or_create(page);
}

template<typename T>
std::string vec_to_str(const std::vector<T>& vec) {
    std::string str = "[";
    for (const auto& ele: vec) {
        str += fmt::format("{}", ele);
        if (&ele != &vec.back()) {
            str += ", ";
        }
    }
    str += "]";
    return str;
}

inline void webview_info_add(
    const std::string& page,
    const std::string& group,
    const std::string& entry,
    const std::string& content
) {
    namespace umt = ::umt;
    umt::ObjManager<webview_info::Page>::find(page)->sub(group).sub(entry).get() = content;
}

} // namespace base

#endif /* BASE_DEBUG_DEBUG_HPP */
