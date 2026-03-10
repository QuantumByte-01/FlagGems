import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_equal, to_reference


@pytest.mark.parametrize(
    "shape_r",
    [
        # (N, C*r^2, H, W), r
        ((1, 4, 1, 1), 2),
        ((1, 9, 1, 1), 3),
        ((2, 4, 8, 8), 2),
        ((2, 36, 16, 16), 6),
        ((4, 16, 32, 32), 4),
        ((1, 4, 64, 64), 2),
        ((2, 4, 256, 256), 2),
        ((1, 4, 512, 512), 2),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pixel_shuffle(shape_r, dtype):
    shape, r = shape_r
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.pixel_shuffle(ref_inp, r)
    with flag_gems.use_gems():
        res_out = torch.pixel_shuffle(inp, r)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize(
    "shape_r",
    [
        # (N, C, H*r, W*r), r
        ((1, 1, 2, 2), 2),
        ((1, 1, 3, 3), 3),
        ((2, 3, 16, 16), 2),
        ((4, 4, 32, 32), 4),
        ((1, 1, 64, 64), 2),
        ((2, 3, 256, 256), 2),
        ((1, 1, 512, 512), 2),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pixel_unshuffle(shape_r, dtype):
    shape, r = shape_r
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.pixel_unshuffle(ref_inp, r)
    with flag_gems.use_gems():
        res_out = torch.pixel_unshuffle(inp, r)
    gems_assert_equal(res_out, ref_out)
