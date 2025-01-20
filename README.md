# triton_learning
a repo for Getting Started on Triton
- reference
https://triton-lang.org/main/getting-started/installation.html
https://zhuanlan.zhihu.com/p/12789107689

## Installation
refer to the official website and here i use vllm==0.6(default install triton 3.0.0)
using spack load cuda@12.1
py 3.10
torch 2.4
## Tutorial

here i will not give the detailed readme for each script, as the comment within the code is enough
## debug
```bash
# dumps the IR before every MLIR pass Triton runs, for all kernels.
export MLIR_ENABLE_DUMP=1
# dumps the IR before every pass run over the LLVM IR
export LLVM_IR_ENABLE_DUMP=1
# enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
export TRITON_KERNEL_DUMP=1
# forces to compile kernels regardless of cache hit.
export TRITON_ALWAYS_COMPILE=1
# specifies the directory from which to load the IR/ptx/amdgcn files when TRITON_KERNEL_OVERRIDE is set to 1.
export TRITON_DUMP_DIR=/home/hzeng/prj/triton_learning/dump/


# easy to paste
export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1
export TRITON_KERNEL_DUMP=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_DUMP_DIR=/home/hzeng/prj/triton_learning/dump/

python3 vector_addition.py 2>&1 | tee ../dump/vadd.mlir
```