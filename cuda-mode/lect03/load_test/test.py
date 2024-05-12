import os
from torch.utils.cpp_extension import load

test = load(
    name='test',
    sources=["./test_cpp.cpp"],
    extra_cflags=['-O2'],
    verbose=True
)

print(test)
print(dir(test))
print(test.hello(10))