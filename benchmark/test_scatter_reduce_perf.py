import torch

import flag_gems

from .performance_utils import Benchmark, BenchLevel, generate_tensor_input


class ScatterReduceBenchmark(Benchmark):
    def __init__(self, reduce):
        self.reduce = reduce
        super().__init__(
            op_name=f"scatter_reduce_{reduce}",
            torch_op=self._op,
            dtypes=[torch.float32],
        )

    def _op(self, *args, **kwargs):
        return torch.scatter_reduce(*args, **kwargs)

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        shapes = [
            (1024, 1024),
            (4096, 4096),
            (128, 512, 512),
        ]
        for shape in shapes:
            dim = 1
            L = shape[dim]
            idx_L = L // 4
            idx_shape = list(shape)
            idx_shape[dim] = idx_L
            idx_shape = tuple(idx_shape)
            inp = torch.randn(shape, dtype=cur_dtype, device=flag_gems.device)
            src = torch.randn(idx_shape, dtype=cur_dtype, device=flag_gems.device)
            idx = torch.randint(0, L, idx_shape, dtype=torch.int64, device=flag_gems.device)
            yield inp, dim, idx, src, self.reduce


for _reduce in ["sum", "amax", "amin", "mean"]:
    globals()[f"bench_scatter_reduce_{_reduce}"] = ScatterReduceBenchmark(_reduce)

if __name__ == "__main__":
    for _reduce in ["sum", "amax", "amin", "mean"]:
        globals()[f"bench_scatter_reduce_{_reduce}"].run(print_data=True)
