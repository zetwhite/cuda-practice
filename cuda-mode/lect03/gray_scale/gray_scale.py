import torch, os, math
import torchvision as tv
import torchvision.transforms.functional as tvf
import matplotlib.pyplot as plt
from torchvision import io
from torch.utils.cpp_extension import load


img = io.read_image('puppy.jpg')
print(img.shape) #[3, 1330, 1920]

def show_img(x, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0) #CHW -> HWC 
    plt.imshow(x.cpu(), **kwargs)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#  Load a PyTorch C++ extension just-in-time (JIT)
m = load(name="m", 
     sources=["./gray_scale.cu"],
     with_cuda=True,
     verbose=True)

img_cuda = img.contiguous().cuda()
res = m.rgb_to_grayscale(img_cuda).cpu()

show_img(res, cmap='gray')
plt.savefig('./puppy_greyscaled.jpg')
print(dir(m))