import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["N", "C_out", "OH", "OW"],
)
@triton.jit
def pixel_shuffle_kernel(
    inp_ptr,
    out_ptr,
    N,
    C_out,
    OH,
    OW,
    H,
    W,
    r,
    C_in,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Map output (N, C_out, OH, OW) <- input (N, C_in, H, W)
    where C_in = C_out * r^2, OH = H * r, OW = W * r.

    out[n, c, oh, ow] = inp[n, c * r^2 + (oh % r) * r + (ow % r), oh // r, ow // r]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    offs64 = tl.cast(offs, tl.int64)
    OW64 = tl.cast(OW, tl.int64)
    OH64 = tl.cast(OH, tl.int64)
    C_out64 = tl.cast(C_out, tl.int64)
    r64 = tl.cast(r, tl.int64)
    W64 = tl.cast(W, tl.int64)
    H64 = tl.cast(H, tl.int64)
    C_in64 = tl.cast(C_in, tl.int64)

    ow = offs64 % OW64
    tmp = offs64 // OW64
    oh = tmp % OH64
    tmp = tmp // OH64
    c = tmp % C_out64
    n = tmp // C_out64

    ph = oh % r64
    pw = ow % r64
    ih = oh // r64
    iw = ow // r64
    ic = c * r64 * r64 + ph * r64 + pw

    inp_off = n * C_in64 * H64 * W64 + ic * H64 * W64 + ih * W64 + iw

    val = tl.load(inp_ptr + inp_off, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


@libentry()
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [512, 1024, 2048, 4096]],
    key=["N", "C_in", "H", "W"],
)
@triton.jit
def pixel_unshuffle_kernel(
    inp_ptr,
    out_ptr,
    N,
    C_in,
    H,
    W,
    C_out,
    OH,
    OW,
    r,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Map output (N, C_out, OH, OW) <- input (N, C_in, H, W)
    where C_out = C_in * r^2, OH = H // r, OW = W // r.

    out[n, c_in * r^2 + ph * r + pw, ih, iw] = inp[n, c_in, ih*r+ph, iw*r+pw]
    i.e. out[n, c, oh, ow] = inp[n, c // (r^2), (oh * r) + (c // r % r), (ow * r) + (c % r)]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    offs64 = tl.cast(offs, tl.int64)
    OW64 = tl.cast(OW, tl.int64)
    OH64 = tl.cast(OH, tl.int64)
    C_out64 = tl.cast(C_out, tl.int64)
    r64 = tl.cast(r, tl.int64)
    W64 = tl.cast(W, tl.int64)
    H64 = tl.cast(H, tl.int64)
    C_in64 = tl.cast(C_in, tl.int64)
    r2 = r64 * r64

    ow = offs64 % OW64
    tmp = offs64 // OW64
    oh = tmp % OH64
    tmp = tmp // OH64
    c = tmp % C_out64
    n = tmp // C_out64

    # reverse mapping
    c_in = c // r2
    ph = (c // r64) % r64
    pw = c % r64
    ih = oh * r64 + ph
    iw = ow * r64 + pw

    inp_off = n * C_in64 * H64 * W64 + c_in * H64 * W64 + ih * W64 + iw

    val = tl.load(inp_ptr + inp_off, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


def pixel_shuffle(input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    logger.debug("GEMS PIXEL_SHUFFLE")
    assert input.ndim == 4, "Input must be 4D (N, C, H, W)"
    N, C_in, H, W = input.shape
    r = upscale_factor
    assert C_in % (r * r) == 0, f"C ({C_in}) must be divisible by upscale_factor^2 ({r*r})"
    C_out = C_in // (r * r)
    OH = H * r
    OW = W * r
    inp = input.contiguous()
    out = torch.empty((N, C_out, OH, OW), dtype=inp.dtype, device=inp.device)
    n_elements = N * C_out * OH * OW
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    pixel_shuffle_kernel[grid](inp, out, N, C_out, OH, OW, H, W, r, C_in, n_elements)
    return out


def pixel_unshuffle(input: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    logger.debug("GEMS PIXEL_UNSHUFFLE")
    assert input.ndim == 4, "Input must be 4D (N, C, H, W)"
    N, C_in, H, W = input.shape
    r = downscale_factor
    assert H % r == 0 and W % r == 0, "H and W must be divisible by downscale_factor"
    C_out = C_in * r * r
    OH = H // r
    OW = W // r
    inp = input.contiguous()
    out = torch.empty((N, C_out, OH, OW), dtype=inp.dtype, device=inp.device)
    n_elements = N * C_out * OH * OW
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    pixel_unshuffle_kernel[grid](inp, out, N, C_in, H, W, C_out, OH, OW, r, n_elements)
    return out
