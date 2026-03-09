import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["M", "N"],
)
@triton.jit
def diff_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    inp_stride_row,
    inp_stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    """out[row, col] = inp[row, col+1] - inp[row, col]  for col in [0, N)"""
    pid = tl.program_id(0)
    n_col_blocks = tl.cdiv(N, BLOCK_SIZE)
    row = pid // n_col_blocks
    col_block = pid % n_col_blocks
    offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < M) & (offs < N)

    a = tl.load(
        inp_ptr + row * inp_stride_row + (offs + 1) * inp_stride_col,
        mask=mask,
        other=0.0,
    )
    b = tl.load(
        inp_ptr + row * inp_stride_row + offs * inp_stride_col,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + row * out_stride_row + offs * out_stride_col,
        a - b,
        mask=mask,
    )


def diff(
    input: torch.Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[torch.Tensor] = None,
    append: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS DIFF")

    ndim = input.dim()
    if dim < 0:
        dim = dim + ndim

    # Handle prepend / append
    parts = [p for p in [prepend, input, append] if p is not None]
    inp = torch.cat(parts, dim=dim) if len(parts) > 1 else input.clone()

    for _ in range(n):
        inp = inp.contiguous()
        L = inp.shape[dim]

        # Move target dim to last, reshape to (M, L)
        if dim != ndim - 1:
            inp = inp.movedim(dim, -1)
        inp_shape = inp.shape
        inp_2d = inp.reshape(-1, L)
        M = inp_2d.shape[0]
        N = L - 1

        out_2d = torch.empty((M, N), dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (M * triton.cdiv(N, meta["BLOCK_SIZE"]),)
        diff_kernel[grid](
            inp_2d, out_2d,
            M, N,
            inp_2d.stride(0), inp_2d.stride(1),
            out_2d.stride(0), out_2d.stride(1),
        )

        out_shape = inp_shape[:-1] + (N,)
        inp = out_2d.reshape(out_shape)
        if dim != ndim - 1:
            inp = inp.movedim(-1, dim)

    return inp
