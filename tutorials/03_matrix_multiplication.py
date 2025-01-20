import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_current_target().backend

    
"""
每个program计算C矩阵的一个元素
下面的kernel简化了一些教程源码，例如假设矩阵大小正好可以被block_size整除，
这样在kernel中就不需要考虑边界问题
缓存优化：
    参考https://zhuanlan.zhihu.com/p/12789107689的图
    A矩阵的一行乘B矩阵的所有元素，可以得到C矩阵的一行（假设一行9个元素）
    A矩阵的3行乘B矩阵的3列，也可以得到C矩阵的9个元素
    但是访存量大大减少，这就是分组的概念
"""

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
            ],
    key=['M', 'N', 'K'], # 这个值的变化会带来 调优配置变化
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1. 这里把pid看成一个二维矩阵，pid_m是行，pid_n是列，每一个pid对应一个block
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)        # 计算在行的方向有几个block
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)        # 计算在列的方向有几个block
    
    # 2. 非常重要的设置：GROUP_SIZE_M 是矩阵A在M维度上有多少个 BLOCK_SIZE_M的块, 
    # 单位直接是Block了，而不是矩阵A的行； BLOCK_SIZE_M 的单位，才是矩阵A的行；
    # 所以GROUP_SIZE_M和矩阵A的行之间还隔了一层Block的单位，即Group - Block - row。
    # GROUP_SIZE_M = 8，意味着把矩阵A沿着M的维度分成了以8个BLOCK_SIZE_M为一组的两个Group
    # 然后想象一下，一个group，在M的维度有8个program，在N的维度有num_pid_n个程序，
    # 因此一共GROUP_SIZE_M * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # 第几个group
    group_id = pid // num_pid_in_group
    # 这个group中第一个程序的行id
    first_pid_m = group_id * GROUP_SIZE_M
    # 预防num_pid_m不是GROUP_SIZE_M的整数倍
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    
    # 3. 计算pid_m和pid_n
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 4. 计算pid_m和pid_n对应的block中每个元素的地址
    # offs_am指的是矩阵A在M这个维度上的偏移量
    """
    * * * *
    * * * *
    如果pid_m是1，那offs_am就是上面0 1 ，第二个*对应的block的行索引(M维度)
    """
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)     # 对于在K维度的偏移量，A和B共享
    # 下面这句分别把偏移量转换成列向量和行向量，以便进行广播，最后就得到这个block中每个元素的地址
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 5.主要计算,沿着K的维度以（BLOCK_SIZE_M, BLOCK_SIZE_N）的计算量计算
    accumlator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 5.1. 读取A和B的元素
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # 5.2. 计算
        accumlator += tl.dot(a, b)
        # 5.3. 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumlator.to(tl.float16)
    
    # 6. 计算结果写回c
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, c)
    
def triton_matmul(a, b):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

if __name__ == '__main__':
    seq_len = 2048
    torch.manual_seed(0)
    a = torch.randn(seq_len, seq_len, device='cuda')
    b = torch.randn(seq_len, seq_len, device='cuda')
    triton_output = triton_matmul(a, b)
    torch_output = torch.matmul(a, b)
    
    print(f'The maximum absolute difference between torch and triton is '
          f'{torch.max(torch.abs(torch_output - triton_output))}')
    