import torch

from .performance_utils import GenericBenchmark, generate_tensor_input


class RollBenchmark(GenericBenchmark):
    input_generator = generate_tensor_input

    def set_more_shapes(self):
        return [
            (1024,),
            (1024 * 1024,),
            (4096, 4096),
            (128, 512, 512),
        ]

    def set_more_metrics(self):
        return []


bench_roll = RollBenchmark(
    op_name="roll",
    torch_op=torch.roll,
    dtypes=[torch.float16, torch.float32, torch.bfloat16],
)

if __name__ == "__main__":
    bench_roll.run(print_data=True)
