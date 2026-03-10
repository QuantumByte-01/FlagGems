import logging
from typing import List, Union

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["n_elements"],
)
@triton.jit
def roll_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    shift,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat roll: out[i] = inp[(i - shift) % n_elements]"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    N = tl.cast(n_elements, tl.int64)
    src = (tl.cast(offs, tl.int64) - tl.cast(shift, tl.int64)) % N
    # Python-style modulo (always non-negative)
    src = tl.where(src < 0, src + N, src)

    val = tl.load(inp_ptr + src, mask=mask, other=0)
    tl.store(out_ptr + offs, val, mask=mask)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["M", "L"],
)
@triton.jit
def roll_dim_kernel(
    inp_ptr,
    out_ptr,
    M,
    L,
    K,
    shift,
    inp_stride_m,
    inp_stride_l,
    inp_stride_k,
    out_stride_m,
    out_stride_l,
    out_stride_k,
    BLOCK_SIZE: tl.constexpr,
):
    """Roll along dim: out[m, l, k] = inp[m, (l - shift) % L, k]"""
    pid = tl.program_id(0)
    n_elements = M * L * K
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    offs64 = tl.cast(offs, tl.int64)
    K64 = tl.cast(K, tl.int64)
    L64 = tl.cast(L, tl.int64)

    k = offs64 % K64
    tmp = offs64 // K64
    l = tmp % L64
    m = tmp // L64

    shift64 = tl.cast(shift, tl.int64)
    src_l = (l - shift64) % L64
    src_l = tl.where(src_l < 0, src_l + L64, src_l)

    in_idx = (
        m * tl.cast(inp_stride_m, tl.int64)
        + src_l * tl.cast(inp_stride_l, tl.int64)
        + k * tl.cast(inp_stride_k, tl.int64)
    )
    out_idx = (
        m * tl.cast(out_stride_m, tl.int64)
        + l * tl.cast(out_stride_l, tl.int64)
        + k * tl.cast(out_stride_k, tl.int64)
    )

    val = tl.load(inp_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + out_idx, val, mask=mask)


def roll(
    input: torch.Tensor,
    shifts: Union[int, List[int]],
    dims: Union[int, List[int]] = (),
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    inp = input.contiguous()
    out = torch.empty_like(inp)
    ndim = inp.dim()
    n_elements = inp.numel()

    # No dims specified: flatten, roll, reshape
    if dims == () or dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
        shift = shifts if isinstance(shifts, int) else shifts[0]
        shift = shift % n_elements
        inp_flat = inp.reshape(-1)
        out_flat = out.reshape(-1)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        roll_kernel[grid](inp_flat, out_flat, n_elements, shift)
        return out

    # Normalize shifts and dims to lists
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    # Apply each (shift, dim) sequentially
    cur = inp
    for shift, dim in zip(shifts, dims):
        if dim < 0:
            dim = dim + ndim
        L = cur.shape[dim]
        shift = shift % L

        # Reshape to (M, L, K)
        M = 1
        for i in range(dim):
            M *= cur.shape[i]
        K = 1
        for i in range(dim + 1, ndim):
            K *= cur.shape[i]

        cur_c = cur.contiguous()
        # Collapse to 3D view
        cur_3d = cur_c.reshape(M, L, K)
        out_3d = torch.empty_like(cur_3d)
        n_el = M * L * K
        grid = lambda meta: (triton.cdiv(n_el, meta["BLOCK_SIZE"]),)
        roll_dim_kernel[grid](
            cur_3d, out_3d,
            M, L, K, shift,
            cur_3d.stride(0), cur_3d.stride(1), cur_3d.stride(2),
            out_3d.stride(0), out_3d.stride(1), out_3d.stride(2),
        )
        cur = out_3d.reshape(cur.shape)

    return cur
