#ifndef BASE_PARAM_PARAMETER_HPP
#define BASE_PARAM_PARAMETER_HPP

#include <chrono>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <string>
#include <thread>
#include <variant>

#include "UltraMultiThread/include/umt/umt.hpp"
#include "base/debug/debug.hpp"

namespace base {

using Param = std::variant<bool, int64_t, double, std::string, std::vector<int64_t>>;

std::shared_ptr<Param> create_param(const std::string& name);

std::shared_ptr<Param> find_param(const std::string& name);

void wait_for_param(const std::string& name);

template<class... Ts>
struct Overloaded: Ts... {
    using Ts::operator()...;
};

template<class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

const auto PARAM_VISITOR =
    Overloaded { [](const auto& arg) -> std::string { return fmt::format("{}", arg); },
                 [](const std::vector<int64_t>& arg) -> std::string {
                     return base::vec_to_str<int64_t>(arg);
                 } };

template<typename T>
T get_param(const std::string& name) {
    auto ptr = find_param(name);
    // 找不到 variant 实例
    if (ptr == nullptr) {
        T value = T();
        ::base::print(
            ::base::PrintMode::ERROR,
            "param",
            "get_param() 找不到名为 \"{}\" 的 variant 实例将返回 {}。",
            name,
            PARAM_VISITOR(value)
        );
        return T();
    }
    Param found = *ptr;
    T* res = std::get_if<T>(&found);
    // 查询类型错误
    if (res == nullptr) {
        T value = T();
        ::base::print(
            ::base::PrintMode::ERROR,
            "param",
            "get_param() 查询 \"{}\" 的类型错误，将返回 {}。",
            name,
            PARAM_VISITOR(value)
        );
        return value;
    }
    return *res;
}

class ParameterManager {
public:
    explicit ParameterManager(const std::string& param_file_path):
        param_file_path(param_file_path) {}

private:
    std::string param_file_path;
};

void parameter_run(const std::string& param_file_path);

} // namespace base

#endif /* BASE_PARAM_PARAMETER_HPP */
