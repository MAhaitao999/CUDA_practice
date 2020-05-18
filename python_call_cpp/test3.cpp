#include <pybind11/pybind11.h>
#include "cuda_test.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "a function which adds two numbers",
          py::arg("i"), py::arg("j"));
    m.def("cuda", &cuda, "testing",
          py::arg("a"), py::arg("b")); // 新增的
}


