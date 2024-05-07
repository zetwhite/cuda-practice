#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

constexpr unsigned int cdiv(unsigned int a, unsigned int b)
{
    return (a + b -1) / b;
}


__global__ void matAddKernel(float* A, float*B, float*C, int m, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int idx = i * n + j;
    printf("idx = %d\n", idx);

    if( i < m && j < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}


void matAdd(float* A, float* B, float* C, int m, int n)
{
    float *A_d, *B_d, *C_d;
    size_t size = m * n * sizeof(float);

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    printf("A[0][0] = %f\n", A[0]);
    printf("B[0][0] = %f\n", B[0]);
    constexpr unsigned int num_thread = 256;
    unsigned int num_block_1 = cdiv(m, num_thread);
    unsigned int num_block_2 = cdiv(n, num_thread); 
    printf("%d %d\n", num_block_1, num_block_2);

    dim3 threads_for_block(num_thread, num_thread, 1);
    dim3 blocks_for_grid(num_block_1, num_block_2, 1);
    
    printf("%d %d\n", blocks_for_grid.x, blocks_for_grid.y);
    matAddKernel<<<blocks_for_grid, threads_for_block>>>(A_d, B_d, C_d, m, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    printf("C[0][0] = %f\n", C[0]);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    printf("C[0][0] = %f\n", C[0]);
}

int main()
{
    const int m = 512;
    const int n = 512;
    float A[m][n];
    float B[m][n];
    float C_cuda[m][n];
    float C_cpu[m][n];


    // generate som  dummy vectors to add
    for (int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            A[i][j] = float(rand() / 10.0f);
            B[i][j] = float(rand() / 10.0f);
        }
    }

    matAdd((float*)A, (float*)B, (float*)C_cuda, m, n);

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            C_cpu[i][j] = A[i][j] + B[i][j];
        }
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(! (C_cpu[i][j] - 0.01f <= C_cuda[i][j]) && (C_cuda[i][j] <= C_cpu[i][j] + 0.01f) )
            {
                printf("%f != %f\n", C_cuda[i][j], C_cpu[i][j]);
                printf("ERROR!\n");
                return 0; 
            }
        }
    }
    printf("CUDA kernel runs correctly!\n");

    return 0;
}