#include <torch/extension.h>
#include <iostream>

int placeholder(int x) {
    std::cout << "hello "<< x << '\n';
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hello", &placeholder, "test");
}