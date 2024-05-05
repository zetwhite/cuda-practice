import torch 

def time_pytorch_function(func, input):
    # wrapper of cuda event
    # a synchronization markers that can be used to monitor the devices
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for _ in range(5):
        func(input)
    
    # record the event in a given stream
    start.record()
    func(input)
    end.record()
    
    # wait for the event to complete
    torch.cuda.synchronize()
    # return elapsed time in ms
    return start.elapsed_time(end)

a = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a*a

def square_3(a):
    return a ** 2

print(time_pytorch_function(torch.square,a))
print(time_pytorch_function(square_2, a))
print(time_pytorch_function(square_3, a))

# 1.462272047996521
# 1.4612480401992798
# 2.1678080558776855


# torch profile doc : https://pytorch.org/tutorials/beginner/profiler.html 
print("=============")
print("Profiling torch.square")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(a)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# =============
# Profiling torch.square
# =============
# STAGE:2024-05-01 22:21:43 31602:31602 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
# STAGE:2024-05-01 22:21:43 31602:31602 ActivityProfilerController.cpp:320] Completed Stage: Collection
# STAGE:2024-05-01 22:21:43 31602:31602 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#              aten::square         0.79%      12.000us         7.03%     107.000us     107.000us      13.000us         0.82%       1.576ms       1.576ms             1  
#                 aten::pow         4.40%      67.000us         5.98%      91.000us      91.000us       1.552ms        98.48%       1.563ms       1.563ms             1  
#                  aten::to         0.00%       0.000us         0.00%       0.000us       0.000us       6.000us         0.38%       6.000us       6.000us             1  
#         aten::result_type         0.07%       1.000us         0.07%       1.000us       1.000us       5.000us         0.32%       5.000us       5.000us             1  
#           cudaEventRecord         1.05%      16.000us         1.05%      16.000us       2.000us       0.000us         0.00%       0.000us       0.000us             8  
#          cudaLaunchKernel         1.25%      19.000us         1.25%      19.000us      19.000us       0.000us         0.00%       0.000us       0.000us             1  
#     cudaDeviceSynchronize        92.44%       1.406ms        92.44%       1.406ms       1.406ms       0.000us         0.00%       0.000us       0.000us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 1.521ms
# Self CUDA time total: 1.576ms


print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(a)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# =============
# Profiling a * a
# =============
# STAGE:2024-05-01 22:21:43 31602:31602 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
# STAGE:2024-05-01 22:21:43 31602:31602 ActivityProfilerController.cpp:320] Completed Stage: Collection
# STAGE:2024-05-01 22:21:43 31602:31602 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                 aten::mul         1.41%      21.000us         2.08%      31.000us      31.000us       1.503ms       100.00%       1.503ms       1.503ms             1  
#           cudaEventRecord         0.40%       6.000us         0.40%       6.000us       3.000us       0.000us         0.00%       0.000us       0.000us             2  
#          cudaLaunchKernel         0.67%      10.000us         0.67%      10.000us      10.000us       0.000us         0.00%       0.000us       0.000us             1  
#     cudaDeviceSynchronize        97.52%       1.452ms        97.52%       1.452ms       1.452ms       0.000us         0.00%       0.000us       0.000us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 1.489ms
# Self CUDA time total: 1.503ms


print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(a)
    
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# =============
# Profiling a ** 2
# =============
# STAGE:2024-05-01 22:22:43 31930:31930 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
# STAGE:2024-05-01 22:22:43 31930:31930 ActivityProfilerController.cpp:320] Completed Stage: Collection
# STAGE:2024-05-01 22:22:43 31930:31930 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                 aten::pow         1.86%      28.000us         2.59%      39.000us      39.000us       1.509ms        99.54%       1.516ms       1.516ms             1  
#         aten::result_type         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.26%       4.000us       4.000us             1  
#                  aten::to         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.20%       3.000us       3.000us             1  
#           cudaEventRecord         0.60%       9.000us         0.60%       9.000us       1.500us       0.000us         0.00%       0.000us       0.000us             6  
#          cudaLaunchKernel         0.47%       7.000us         0.47%       7.000us       7.000us       0.000us         0.00%       0.000us       0.000us             1  
#     cudaDeviceSynchronize        97.07%       1.460ms        97.07%       1.460ms       1.460ms       0.000us         0.00%       0.000us       0.000us             1  
# -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 1.504ms
# Self CUDA time total: 1.516ms

