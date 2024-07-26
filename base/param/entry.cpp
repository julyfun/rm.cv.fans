#include <iostream>

#include "UltraMultiThread/include/umt/umt.hpp"
#include "base/param/parameter.hpp"

namespace base {
void background_parameter_run(const std::string& param_file_path) {
    std::thread([=]() { base::parameter_run(param_file_path); }).detach();
}
} // namespace base

PYBIND11_EMBEDDED_MODULE(base_param, m) {
    namespace py = pybind11;
    m.def("background_parameter_run", base::background_parameter_run, py::arg("param_file_path"));
}
