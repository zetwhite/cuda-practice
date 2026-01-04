#include <cuda.h> 
#include <iostream> 

const int H = 20; 
const int W = 10; 
const int C = 3; 

__global__ void quant(float* tensor_dev, uint8_t* tensor_dev_q, float scale, int zerop)
{
    int h = threadIdx.x + blockDim.x * blockIdx.x; 
    int w = threadIdx.y + blockDim.y * blockIdx.y; 
    int c = threadIdx.z + blockDim.z * blockIdx.z; 

    // R = S(Q - Z)
    // Q = round(R/S) + Z
    if(h < H && w < W && c < C)
    {
        int idx = (h * W * C) + (w * C) + c; 
        int q = (int) nearbyintf(tensor_dev[idx] / scale) + zerop;

        if(q > 255)
            q = 255; 

        else if (q < 0)
            q = 0; 

        tensor_dev_q[idx] = q; 
    }
    
}


constexpr unsigned int cdiv(unsigned int a, unsigned int b)
{
    return (a + b -1) / b;
}


// Function to generate a random float between a specified min and max
float randomFloat(float min, float max) {
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float range = max - min;
    return min + (random * range);
}

int main() 
{
    float tensor[H][W][C]; 
    uint8_t tensor_q[H][W][C];

    for(int i = 0; i < H; i++)
    {
        for(int j = 0; j < W; j++)
        {
            for(int k = 0; k < C; k++)
            {
                tensor[i][j][k] = randomFloat(-10.0, 10.0);
            }
        }
    }


    float *tensor_dev; 
    size_t size = H * W * C * sizeof(float); 
    cudaMalloc((void**)&tensor_dev, size); 

    uint8_t *tensor_dev_q;
    size_t size_q = H * W * C * sizeof(int8_t); 
    cudaMalloc((void**)&tensor_dev_q, size_q);

    float scale = 20.0 / 255;
    int zerop = 0; 

    cudaMemcpy(/* dest */tensor_dev, /* src*/tensor, size, cudaMemcpyHostToDevice); 

    dim3 threads_for_block(16, 16, 4); 
    dim3 blocks_for_grid(cdiv(H, 16), cdiv(W, 16), cdiv(C, 4)); 

    quant<<<blocks_for_grid, threads_for_block>>>(tensor_dev, tensor_dev_q, scale, zerop);
    cudaDeviceSynchronize();

    cudaMemcpy(/* dest */tensor_q, /* src */ tensor_dev_q, size_q, cudaMemcpyDeviceToHost);

    cudaFree(tensor_dev); 
    cudaFree(tensor_dev_q);

    for(int i = 0; i < H; i++)
        for(int j = 0; j < W; j++)
            for(int k = 0; k < C; k++)
                std::cout << int(tensor_q[i][j][k]) << std::endl; 
    return 0;
}