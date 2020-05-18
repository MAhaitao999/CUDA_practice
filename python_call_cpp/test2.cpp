#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int add(py::array_t<float> &array, int col) {
    
    py::buffer_info buf1 = array.request();
    float *p = (float *)buf1.ptr;
    for (int i = 0; i < col; i++) {
        printf("cur value %lf\n", *p++);
    }
    printf("the p is %p\n", p);

    return 0;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "a function which adds two numbers");
}

