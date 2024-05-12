## notes 

###  LibPytorch installation

> We provide binary distributions of all headers, libraries and CMake configuration files required to depend on PyTorch. We call this distribution LibTorch, ...

* https://pytorch.org/cppdocs/installing.html


### Custom C++ & CUDA extensions

> C++ extensions are a mechanism we have developed to allow users (you) to create PyTorch operators defined out-of-source, i.e. separate from the PyTorch backend. 

* https://pytorch.org/tutorials/advanced/cpp_extension.html 

* update library path to include `libtorch/include`. 
<
    * for example, vscode update `c_cpp_properities.json` : 
        ```json 
        {
            "configurations": [
                {
                    "name": "Linux",
                    "includePath": [
                        "${workspaceFolder}/**",
                        "/usr/lib/libtorch/include"
                    ],
                    "defines": [],
                    "compilerPath": "/usr/bin/gcc",
                    "cStandard": "c17",
                    "cppStandard": "gnu++14",
                    "intelliSenseMode": "linux-gcc-x64"
                }
            ],
            "version": 4
        }
        ```