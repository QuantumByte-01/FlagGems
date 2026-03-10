import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["M", "IDX_L", "K"],
)
@triton.jit
def scatter_reduce_sum_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    M,
    IDX_L,
    K,
    OUT_L,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    offs64 = tl.cast(offs, tl.int64)
    K64 = tl.cast(K, tl.int64)
    IDX_L64 = tl.cast(IDX_L, tl.int64)
    OUT_L64 = tl.cast(OUT_L, tl.int64)

    k = offs64 % K64
    tmp = offs64 // K64
    i = tmp % IDX_L64
    m = tmp // IDX_L64

    src_off = m * IDX_L64 * K64 + i * K64 + k
    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    idx = tl.load(idx_ptr + src_off, mask=mask, other=0).to(tl.int64)

    dst_off = m * OUT_L64 * K64 + idx * K64 + k
    tl.atomic_add(out_ptr + dst_off, val, mask=mask)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["M", "IDX_L", "K"],
)
@triton.jit
def scatter_reduce_count_kernel(
    idx_ptr,
    cnt_ptr,
    M,
    IDX_L,
    K,
    OUT_L,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Increment count for each scattered position (for 'mean' reduce)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    offs64 = tl.cast(offs, tl.int64)
    K64 = tl.cast(K, tl.int64)
    IDX_L64 = tl.cast(IDX_L, tl.int64)
    OUT_L64 = tl.cast(OUT_L, tl.int64)

    k = offs64 % K64
    tmp = offs64 // K64
    i = tmp % IDX_L64
    m = tmp // IDX_L64

    src_off = m * IDX_L64 * K64 + i * K64 + k
    idx = tl.load(idx_ptr + src_off, mask=mask, other=0).to(tl.int64)

    dst_off = m * OUT_L64 * K64 + idx * K64 + k
    tl.atomic_add(cnt_ptr + dst_off, 1, mask=mask)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["M", "IDX_L", "K"],
)
@triton.jit
def scatter_reduce_amax_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    M,
    IDX_L,
    K,
    OUT_L,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    offs64 = tl.cast(offs, tl.int64)
    K64 = tl.cast(K, tl.int64)
    IDX_L64 = tl.cast(IDX_L, tl.int64)
    OUT_L64 = tl.cast(OUT_L, tl.int64)

    k = offs64 % K64
    tmp = offs64 // K64
    i = tmp % IDX_L64
    m = tmp // IDX_L64

    src_off = m * IDX_L64 * K64 + i * K64 + k
    val = tl.load(src_ptr + src_off, mask=mask, other=float("-inf"))
    idx = tl.load(idx_ptr + src_off, mask=mask, other=0).to(tl.int64)

    dst_off = m * OUT_L64 * K64 + idx * K64 + k
    tl.atomic_max(out_ptr + dst_off, val, mask=mask)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["M", "IDX_L", "K"],
)
@triton.jit
def scatter_reduce_amin_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    M,
    IDX_L,
    K,
    OUT_L,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    offs64 = tl.cast(offs, tl.int64)
    K64 = tl.cast(K, tl.int64)
    IDX_L64 = tl.cast(IDX_L, tl.int64)
    OUT_L64 = tl.cast(OUT_L, tl.int64)

    k = offs64 % K64
    tmp = offs64 // K64
    i = tmp % IDX_L64
    m = tmp // IDX_L64

    src_off = m * IDX_L64 * K64 + i * K64 + k
    val = tl.load(src_ptr + src_off, mask=mask, other=float("inf"))
    idx = tl.load(idx_ptr + src_off, mask=mask, other=0).to(tl.int64)

    dst_off = m * OUT_L64 * K64 + idx * K64 + k
    tl.atomic_min(out_ptr + dst_off, val, mask=mask)


def scatter_reduce(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    logger.debug("GEMS SCATTER_REDUCE")
    ndim = input.ndim
    if dim < 0:
        dim = dim + ndim

    # Compute M, OUT_L, K for 3D reshape
    M = 1
    for i in range(dim):
        M *= input.shape[i]
    OUT_L = input.shape[dim]
    K = 1
    for i in range(dim + 1, ndim):
        K *= input.shape[i]
    IDX_L = index.shape[dim]
    N = M * IDX_L * K

    # Upcast float16/bfloat16 to float32 for atomic operations
    orig_dtype = input.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        input_f = input.float()
        src_f = src.float()
    else:
        input_f = input
        src_f = src

    # Initialize output based on reduce type and include_self
    if include_self:
        out = input_f.clone()
    else:
        if reduce in ("sum", "mean"):
            out = torch.zeros_like(input_f)
        elif reduce == "prod":
            out = torch.ones_like(input_f)
        elif reduce == "amax":
            out = torch.full_like(input_f, float("-inf"))
        elif reduce == "amin":
            out = torch.full_like(input_f, float("inf"))
        else:
            raise ValueError(f"Unsupported reduce: {reduce}")

    # Reshape for 3D kernel
    src_3d = src_f.contiguous().reshape(M, IDX_L, K)
    idx_3d = index.contiguous().reshape(M, IDX_L, K)
    out_3d = out.reshape(M, OUT_L, K)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    if reduce == "sum":
        scatter_reduce_sum_kernel[grid](src_3d, idx_3d, out_3d, M, IDX_L, K, OUT_L, N)

    elif reduce == "mean":
        scatter_reduce_sum_kernel[grid](src_3d, idx_3d, out_3d, M, IDX_L, K, OUT_L, N)
        # Count scattered elements per output position
        if include_self:
            # Each output position already counts the input value
            count = torch.ones(M, OUT_L, K, dtype=torch.int32, device=input.device)
        else:
            count = torch.zeros(M, OUT_L, K, dtype=torch.int32, device=input.device)
        scatter_reduce_count_kernel[grid](idx_3d, count, M, IDX_L, K, OUT_L, N)
        out_3d.div_(count.clamp(min=1).to(out_3d.dtype))

    elif reduce == "amax":
        scatter_reduce_amax_kernel[grid](src_3d, idx_3d, out_3d, M, IDX_L, K, OUT_L, N)

    elif reduce == "amin":
        scatter_reduce_amin_kernel[grid](src_3d, idx_3d, out_3d, M, IDX_L, K, OUT_L, N)

    elif reduce == "prod":
        # Use PyTorch scatter_ multiply as fallback (correct for all dtypes)
        out.scatter_(dim, index, src_f, reduce="multiply")

    result = out.reshape(input.shape)
    if orig_dtype in (torch.float16, torch.bfloat16):
        result = result.to(orig_dtype)
    return result
