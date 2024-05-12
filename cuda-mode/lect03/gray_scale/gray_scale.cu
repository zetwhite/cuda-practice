#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        out[i] = 0.2989*x[i] + 0.5870*x[i+n] +0.1140*x[i+2*n]; 
    }
}

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(1); // input shape is (c, h, w)
    int w = input.size(2);
    printf("h*w: %d*%d\n", h, w);

    torch::Tensor output = torch::empty({h,w}, input.options());
    int threads = 256;

    rgb_to_grayscale_kernel<<<cdiv(w*h,threads), threads>>>(
        input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rgb_to_grayscale", torch::wrap_pybind_function(rgb_to_grayscale), "rgb to grayscale");
}