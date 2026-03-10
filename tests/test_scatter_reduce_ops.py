import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference


REDUCE_TYPES = ["sum", "prod", "mean", "amax", "amin"]


@pytest.mark.parametrize(
    "shape", [(64,), (256, 256), (32, 64, 64), (4, 8, 32, 32)]
)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("reduce", REDUCE_TYPES)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_scatter_reduce(shape, dim, reduce, include_self, dtype):
    ndim = len(shape)
    actual_dim = dim % ndim
    # skip invalid dim
    if actual_dim >= ndim:
        pytest.skip("dim out of range")

    # Create small index tensor (same shape as input, values in [0, L))
    L = shape[actual_dim]
    idx_shape = list(shape)
    # Use a smaller index range for meaningful reduction
    idx_shape[actual_dim] = max(1, L // 2)
    idx_shape = tuple(idx_shape)

    src = torch.randn(idx_shape, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, L, idx_shape, dtype=torch.int64, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = index.cpu()

    ref_out = torch.scatter_reduce(ref_inp, actual_dim, ref_index, ref_src, reduce=reduce, include_self=include_self)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, actual_dim, index, src, reduce=reduce, include_self=include_self)

    gems_assert_close(res_out, ref_out, dtype, atol=1e-4)


@pytest.mark.parametrize("shape", [(1024, 1024), (4096, 4096)])
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_reduce_large(shape, reduce, dtype):
    dim = 1
    L = shape[1]
    idx_shape = (shape[0], L // 4)
    src = torch.randn(idx_shape, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, L, idx_shape, dtype=torch.int64, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = index.cpu()

    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, reduce=reduce, include_self=True)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, reduce=reduce, include_self=True)

    gems_assert_close(res_out, ref_out, dtype)
