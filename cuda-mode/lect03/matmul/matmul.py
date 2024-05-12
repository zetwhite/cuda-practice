import gzip,pickle
import os
from urllib.request import urlretrieve
from pathlib import Path
from torch import tensor
import torch 

from torch.utils.cpp_extension import load

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'

if not os.path.exists('./data'):
    MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
    path_data.mkdir(exist_ok=True)

if not path_gz.exists(): 
    urlretrieve(MNIST_URL, path_gz)

with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))

# print(x_train.shape,x_train.type()) 
# [50000, 784], torch.FloatTensor

if not os.path.exists("./tmp"):
    os.mkdir("./tmp")    

m = load(name = "m", 
         sources=["./matmul.cu"],
         with_cuda = True,
         verbose = True, 
         build_directory="./tmp" 
        )

# simple test 
# a = torch.rand((5, 10)).cuda()
# b = torch.rand((10, 6)).cuda()
# c = m.matmul(a, b).cpu() 

# print(c.shape)
# print(c)


mat1 = x_train.contiguous()
mat2 = torch.randn(784, 10).contiguous()

print(mat1.shape)
print(mat2.shape)

mat1_cuda = mat1.cuda()
mat2_cuda = mat2.cuda()

res = mat1@mat2
res_cuda = m.matmul(mat1_cuda, mat2_cuda).cpu()

ret = torch.allclose(res_cuda, res, atol=1e-3)

print(res.shape)
print(res_cuda.shape)
print(res[0])
print(res_cuda[0])

if ret == True:
    print("Success")
else:
    print("Fail")