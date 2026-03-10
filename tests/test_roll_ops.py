import pytest
import torch

import flag_gems

from .accuracy_utils import ALL_FLOAT_DTYPES, INT_DTYPES, gems_assert_equal, to_reference


@pytest.mark.parametrize(
    "shape", [(1024,), (1024, 1024), (4, 256, 256), (2, 4, 64, 64)]
)
@pytest.mark.parametrize("shift", [0, 1, 13, -7, 512])
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + list(INT_DTYPES))
def test_accuracy_roll_no_dim(shape, shift, dtype):
    if dtype in (torch.float16, torch.bfloat16):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype == torch.float32 or dtype == torch.float64:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.roll(ref_inp, shift)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shift)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", [(1024,), (1024, 1024), (4, 256, 256)])
@pytest.mark.parametrize(
    "shift_dim", [(3, 0), (-7, 1), (512, -1), (0, 0)]
)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + list(INT_DTYPES))
def test_accuracy_roll_with_dim(shape, shift_dim, dtype):
    shift, dim = shift_dim
    ndim = len(shape)
    if dim >= ndim or dim < -ndim:
        pytest.skip("dim out of range for shape")

    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.roll(ref_inp, shift, dim)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shift, dim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", [(64, 128, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_accuracy_roll_multi_dim(shape, dtype):
    shifts = [3, -7, 13]
    dims = [0, 1, 2]
    if dtype == torch.float32:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    gems_assert_equal(res_out, ref_out)
