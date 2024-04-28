#include "base/param/parameter.hpp"

#include <chrono>
#include <cstddef>
#include <memory>

#include "UltraMultiThread/include/umt/ObjManager.hpp"
#include "base/debug/debug.hpp"
#include "third_party/tomlplusplus/toml.hpp"

namespace base {
const std::string PARAM_PREFIX = "base.param.";

Param get_value(const toml::node& node) {
    if (node.is_boolean()) {
        return (*node.as_boolean()).get();
    }
    if (node.is_integer()) {
        return (*node.as_integer()).get();
    }
    if (node.is_floating_point()) {
        return (*node.as_floating_point()).get();
    }
    if (node.is_string()) {
        return (*node.as_string()).get();
    }
    if (node.is_array() && (node.as_array()->is_homogeneous<int64_t>() || node.as_array()->empty()))
    {
        std::vector<int64_t> vec_int;
        for (const auto& ele: *node.as_array()) {
            vec_int.push_back((*ele.as_integer()).get());
        }
        return vec_int;
    }
    return Param();
}

std::shared_ptr<Param> create_param(const std::string& name) {
    namespace umt = ::umt;
    return umt::ObjManager<Param>::create(PARAM_PREFIX + name);
}

std::shared_ptr<Param> find_param(const std::string& name) {
    namespace umt = ::umt;
    return umt::ObjManager<Param>::find(PARAM_PREFIX + name);
}

void wait_for_param(const std::string& name) {
    while (base::find_param(name) == nullptr) {
        using namespace std::chrono_literals;
        ::base::print(::base::PrintMode::INFO, "param", "正在等待参数创建完毕的信号。");
        std::this_thread::sleep_for(200ms);
    }
}

class ParamManager {
public:
    void load_and_update(const std::string& param_file_path);

private:
    void parse(const toml::node& node, const std::string& prefix);

    bool init_ok = false;
    std::set<std::shared_ptr<Param>> param_set;
};

void ParamManager::load_and_update(const std::string& param_file_path) {
    using namespace std::chrono_literals;
    while (true) {
        try {
            const auto table =
                toml::parse_file(std::string(CMAKE_DEF_PROJECT_DIR) + "/" + param_file_path);
            parse(table, "");
            if (!this->init_ok) {
                this->init_ok = true;
                this->param_set.emplace(create_param("ok"));
                ::base::print(::base::PrintMode::INFO, "param", "参数创建完了..");
            }
        } catch (const std::exception& e) {
            ::base::print(::base::PrintMode::ERROR, "param", "{}", e.what());
        }
        std::this_thread::sleep_for(1s);
    }
}

void ParamManager::parse(const toml::node& node, const std::string& prefix) {
    if (node.is_table()) {
        for (const auto& child: *node.as_table()) {
            parse(
                child.second,
                (prefix == "" ? "" : prefix + ".") + std::string(child.first.str())
            );
        }
    } else {
        const Param res = get_value(node);
        const auto found = find_param(prefix);
        if (found == nullptr) {
            const auto ptr = create_param(prefix);
            this->param_set.emplace(ptr);
            *ptr = res;
        } else {
            if (*found != res) {
                const auto tmp = *found;
                *found = res;
                ::base::print(
                    ::base::PrintMode::INFO,
                    "param",
                    "参数 {} 修改: {} -> {}",
                    prefix,
                    std::visit(PARAM_VISITOR, tmp),
                    std::visit(PARAM_VISITOR, res)
                );
            }
        }
    }
}

void parameter_run(const std::string& param_file_path) {
    using namespace std::chrono_literals;

    ParamManager manager;
    manager.load_and_update(param_file_path);
}

} // namespace base
