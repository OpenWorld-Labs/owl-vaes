import torch.nn.functional as F
from torch import Tensor, nn

from ..configs import TransformerConfig
from .attn import StackedTransformer


class CRT(nn.Module):
    def __init__(self, dim, config: TransformerConfig | None = None):
        super().__init__()

        if config is None:
            config = TransformerConfig(
                n_layers=6,
                n_heads=3,
                d_model=384,
                causal=True
            )

        self.core = StackedTransformer(config)
        self.proj_in = nn.Linear(dim, config.d_model, bias = False)
        self.proj_out = nn.Linear(config.d_model, dim, bias = False)


    def forward(self, x: Tensor):
        # x is [b,n,d]
        x_in = x[:,:-1]
        x_out = x[:,1:]

        x_in = self.proj_in(x_in)
        pred = self.core(x_in)
        pred = self.proj_out(pred)
        return F.mse_loss(pred, x_out)

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crt = CRT(768).bfloat16().to(device)
    x = torch.randn(1,16,768).bfloat16().to(device)

    with torch.no_grad():
        print(crt(x))