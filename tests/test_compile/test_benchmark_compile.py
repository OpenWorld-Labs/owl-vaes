from torch.utils.benchmark import Timer
from functools import partial

@torch.compile()
def test_function_compiled(
    x: torch.Tensor,
) -> torch.Tensor:
    return test_function(x)

batch_size, seq_len, num_heads, head_dim, rope_dim = 32, 1024, 8, 64, 48

for dtype in [torch.bfloat16, torch.float32]:
    torch.manual_seed(42)
    x = # torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)

    for f in [test_function, test_function_compiled]:
        timer = Timer("f()", globals={"f": partial(f, x, (cos, sin))}, label=f.__name__, sub_label=str(dtype))
        measurement = timer.blocked_autorange(min_run_time=1)
        print(str(measurement).split("\n", 1)[1], end="\n\n")