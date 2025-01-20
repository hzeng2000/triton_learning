import torch

import triton
import triton.language as tl
from triton.runtime import driver

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = triton.runtime.driver.active.get_current_target().backend

# below is a hack to check if we are running on ROCm
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == 'hip'

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 
                                                                                   'gfx90a', 'gfx908')
def naive_softmax(x):
    """Compute row-wise softmax of x using native PyTorch
    substract max to avoid overflow

    Args:
        x (_type_): 2-dim tensor
    """
    # read MN elements ; write M elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read MN elements ; write MN elements
    numerator = torch.exp(z)
    # read MN elements ; write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; write 3MN + 2M elements
    return ret


"""
here we can see the naively custom implementation of softmax totally requires 5MN + 2M reads and 3MN + 2M writes from and to DRAM. 
This is obvisously wasteful.
we prefer do once read and write to DRAM, and do all the computation on-chip.
the ideal speed-up is 4x(i.e., (8MN+4M)/2MN)
The torch.jit.script flags aims to perform this kind of “kernel fusion” automatically but, as we will see later, it is still far from ideal
"""

"""triton softmax kernel

the kernel parallelizes over the rows of the input
triton website newest code is hard to read and i can not run, this is a simpler version for freshman
triton is so much close to cuda programing
in cuda, the smallest unit is thread, and in triton is block
"""
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, 
                   BLOCK_SIZE: tl.constexpr):
    # similar to cuda gird id
    row_idx = tl.program_id(axis=0)
    # 每个block负责一行的计算，得到起始行的地址
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 得到这一行所有元素的起始地址
    input_ptrs = row_start_ptr + col_offsets
    # 加pad，因为block-size不一定正好和n_col一样
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    # triton的exp操作和cuda一样 快但是不一定准
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
        
# 总的来说，按照cuda编程来说，上面的除了load和store显式使用global memory，其余如果cuda要用shared memory确实麻烦很多
# help function
def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[n_rows, ](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE)
    return y
    
# unit test
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

# benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2 ** k for k in range(2, 20)],
        line_arg='provider',
        line_vals=['triton', 'torch-naive', 'torch-jit'],
        line_names=['triton', 'torch-naive', 'torch-jit'],
        styles=[('blue', '-'),('green', '-'),('green', '--')],
        ylabel='GB/s',
        plot_name='softmax-performance',
        args={'M':4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    elif provider == 'torch-naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=True, print_data=True)