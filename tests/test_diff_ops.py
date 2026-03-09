import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.parametrize("shape", [(1024,), (128, 1024), (4, 64, 256), (2, 3, 64, 64)])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_diff(shape, n, dim, dtype):
    ndim = len(shape)
    actual_dim = dim if dim >= 0 else dim + ndim
    if shape[actual_dim] <= n:
        pytest.skip("shape too small for n diffs")
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(x)
    ref_out = torch.diff(ref_inp, n=n, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.diff(x, n=n, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(1024,), (128, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_diff_prepend_append(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    pre = torch.zeros((1,) if len(shape) == 1 else (1, shape[1]), dtype=dtype, device="cuda")
    app = torch.ones((1,) if len(shape) == 1 else (1, shape[1]), dtype=dtype, device="cuda")
    ref_out = torch.diff(to_reference(x), n=1, dim=0, prepend=to_reference(pre), append=to_reference(app))
    with flag_gems.use_gems():
        res_out = torch.diff(x, n=1, dim=0, prepend=pre, append=app)
    gems_assert_close(res_out, ref_out, dtype)
