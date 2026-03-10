import torch

from .performance_utils import GenericBenchmark, generate_tensor_input


def pixel_shuffle_input_gen(shape, dtype, device):
    # shape = (N, C_out, H*r, W*r) but we need (N, C_out*r^2, H, W)
    # We use shape as (N, C*r^2, H, W) with r=2 baked in
    return torch.randn(shape, dtype=dtype, device=device)


class PixelShuffleBenchmark(GenericBenchmark):
    input_generator = pixel_shuffle_input_gen

    def set_more_shapes(self):
        # (N, C*r^2, H, W) with r=2
        return [
            (1, 4, 64, 64),
            (1, 4, 256, 256),
            (2, 4, 512, 512),
            (4, 4, 256, 256),
            (1, 4, 1024, 1024),
        ]

    def set_more_metrics(self):
        return []


bench_pixel_shuffle = PixelShuffleBenchmark(
    op_name="pixel_shuffle",
    torch_op=lambda x: torch.pixel_shuffle(x, 2),
    dtypes=[torch.float16, torch.float32, torch.bfloat16],
)

bench_pixel_unshuffle = PixelShuffleBenchmark(
    op_name="pixel_unshuffle",
    torch_op=lambda x: torch.pixel_unshuffle(x, 2),
    dtypes=[torch.float16, torch.float32, torch.bfloat16],
)

if __name__ == "__main__":
    bench_pixel_shuffle.run(print_data=True)
    bench_pixel_unshuffle.run(print_data=True)
