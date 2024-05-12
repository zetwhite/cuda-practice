#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTINUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTINUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1)/b; 
}

__global__ void matmul_kernel(float* m, float*n, float* out, int r, int c, int k)
{
    int r_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int c_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if(r_idx < r && c_idx < c)
    {
        float o = 0; 
        for(int i = 0; i < k; i++)
        {
            o += m[ r_idx*k + i ] * n[ i*c + c_idx ];
        }
        out[c * r_idx + c_idx] = o;
    }
    return;   
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n)
{
    CHECK_INPUT(m);
    CHECK_INPUT(n); 

    int r = m.size(0);
    int k = m.size(1); 
    int c = n.size(1);

    assert(m.size(1) == n.size(0));  

    printf("m_row, m_cow : %d, %d\n", r, k);
    printf("n_row, n_col : %d, %d\n", k, c);

    torch::Tensor output = torch::zeros({r, c}, m.options());
    
    dim3 thread_per_block(16, 16); 
    dim3 blocks(cdiv(r, thread_per_block.x), cdiv(c, thread_per_block.y)); 
    
    matmul_kernel<<<blocks, thread_per_block>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), r, c, k
    ); 
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", torch::wrap_pybind_function(matmul), "matmul");
}

/* 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rgb_to_grayscale", torch::wrap_pybind_function(rgb_to_grayscale), "rgb to grayscale");
}
*/