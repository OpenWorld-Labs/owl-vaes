import torch
from torch import nn
import torch.nn.functional as F
import einops as eo
import math

from copy import deepcopy

from ..utils import freeze
from ..configs import TransformerConfig
from ..nn.resnet import SquareToLandscape, LandscapeToSquare

from ..nn.dit import DiT, FinalLayer
from ..nn.embeddings import LearnedPosEnc
from ..nn.embeddings import TimestepEmbedding, StepEmbedding

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

def find_nearest_square(size):
    h,w = size
    avg = (h + w) / 2
    return 2 ** int(round(torch.log2(torch.tensor(avg)).item()))

class DiffusionDecoderCore(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        ch_0 = 16 # To lighten load on initial proj layers

        # Landscape Assumption
        size = config.sample_size
        self.conv_in = LandscapeToSquare(config.channels, ch_0)
        self.conv_out = SquareToLandscape(ch_0, config.channels)
        size = find_nearest_square(size)
        self.original_sample_size = deepcopy(config.sample_size)
        config.sample_size = size

        self.proj_in = nn.Linear(config.patch_size * config.patch_size * ch_0 * 4, config.d_model, bias = False)
        self.pos_enc_x = LearnedPosEnc((32 // config.patch_size)**2, config.d_model)
        self.proj_out = nn.Linear(config.d_model, config.patch_size * config.patch_size * ch_0 * 4, bias = False)

        self.d_embed = StepEmbedding(config.d_model)
        self.ts_embed = TimestepEmbedding(config.d_model)
    
        
        self.proj_in_z = nn.Linear(config.latent_channels, config.d_model)
        self.pos_enc_z = LearnedPosEnc(config.latent_size**2, config.d_model)
        
        self.final = FinalLayer(config, skip_proj = True)

        self.p = config.patch_size
        self.n_p_y = 32 // self.p
        self.n_p_x = 32 // self.p

        self.blocks = DiT(config)
        self.config = config

    @torch.no_grad()
    def sample(self, z):
        d = torch.full((len(z),), 1., device = z.device, dtype = z.dtype)
        ts = torch.full((len(z),), 1., device = z.device, dtype = z.dtype)
        x = torch.randn(len(z), self.config.channels, self.original_sample_size[0], self.original_sample_size[1], device = z.device, dtype=z.dtype)
        return x - self.forward(x, z, ts, d)

    @torch.no_grad()
    def sample4(self, z):
        d  = torch.full((len(z),), 4, device=z.device,dtype=z.dtype)
        ts = torch.full((len(z),), 1., device=z.device,dtype=z.dtype)
        x = torch.randn(len(z), self.config.channels, self.original_sample_size[0], self.original_sample_size[1], device = z.device, dtype=z.dtype)
        dt = .25

        for _ in range(4):
            pred = self.forward(x, z, ts, d)
            x = x - dt * pred
            ts = ts - dt
        
        return x

    def forward(self, x, z, ts, d):
        # x is [b,c,h,w]
        # z is [b,c,h,w] but different size cause latent
        # ts is [b,] in [0,1]
        # d is [b,] in [1,2,4,...,128]

        cond = self.ts_embed(ts) + self.d_embed(d)

        x = self.conv_in(x) # -> [b,d,s,s]
        x = F.pixel_unshuffle(x, downscale_factor=2) # [b,d,2s,2s] -> [b,4d,s,s]
        
        x = eo.rearrange(
            x,
            'b c (n_p_y p_y) (n_p_x p_x) -> b (n_p_y n_p_x) (p_y p_x c)',
            p_y=self.p,p_x=self.p
        )
        x = self.proj_in(x) # -> [b,n,d]
        x = self.pos_enc_x(x)

        z = eo.rearrange(z, 'b c h w -> b (h w) c')
        z = self.proj_in_z(z)
        z = self.pos_enc_z(z)
        
        n = x.shape[1]
        x = torch.cat([x,z],dim=1)

        x = self.blocks(x, cond)
        x = x[:,:n]

        x = self.final(x, cond)
        x = self.proj_out(x)
        x = eo.rearrange(
            x,
            'b (n_p_y n_p_x) (p_y p_x c) -> b c (n_p_y p_y) (n_p_x p_x)',
            p_y = self.p, p_x = self.p,
            n_p_y = self.n_p_y, n_p_x = self.n_p_x
        )

        x = F.pixel_shuffle(x, upscale_factor=2) # 
        x = self.conv_out(x)

        return x

def sample_discrete_timesteps(steps, eps = 1.0e-6):
    # steps is Tensor([1,4,2,64,16]) as an example
    b = len(steps)
    ts = torch.rand(b, device=steps.device, dtype=steps.dtype) * (steps - eps)
    ts = ts.clamp(eps).ceil() / steps
    """
    Example, if d was all 2, ts would be [0,2]
    so do clamp, then ceil will be 1 or 2 (0, 2]
    then do t / 2 and get 0.5 or 1.0, our desired timesteps
    """
    return ts

def sample_steps(b, device, dtype, min_val = 0):
    valid = torch.tensor([2**i for i in range(min_val, 8)]) # [1,2,...,128]
    inds = torch.randint(low=0,high=len(valid), size = (b,))
    steps = valid[inds].to(device=device,dtype=dtype)
    return steps

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = DiffusionDecoderCore(config)
        self.ema = None
        self.sc_frac = 0.25

    def set_ema_core(self, ema):
        if hasattr(ema.ema_model, 'module'):
            self.ema = ema.ema_model.module.core
        else:
            self.ema = ema.ema_model.core
        freeze(self.ema)

    @torch.no_grad()
    #@torch.compile(mode='max-autotune',fullgraph=True)
    def get_sc_targets(self, x, z):
        steps_slow = sample_steps(len(x),x.device,x.dtype, min_val = 1)
        steps_fast = steps_slow / 2

        dt_slow = 1./steps_slow
        dt_fast = 1./steps_fast

        def expand(t):
            return t[:,None,None,None]

        ts = sample_discrete_timesteps(steps_fast)

        pred_1 = self.ema(x, z, ts, steps_slow)
        x_new = x - pred_1 * expand(dt_slow)
        ts_new = ts - dt_slow

        pred_2 = self.ema(x_new, z, ts_new, steps_slow)
        pred = 0.5 * (pred_1 + pred_2)
        return pred, steps_fast, ts

    def get_sc_loss(self, x, z):
        target, steps, ts = self.get_sc_targets(x, z)
        pred = self.core(x, z, ts, steps)
        sc_loss = F.mse_loss(pred, target)
        return sc_loss
    
    def forward(self, x, z):
        b = int(len(x) * (1 - self.sc_frac))
        x,x_sc = x[:b], x[b:]
        z,z_sc = z[:b], z[b:]

        with torch.no_grad():
            steps = sample_steps(len(x),x.device,x.dtype)
            ts = sample_discrete_timesteps(steps)

            eps = torch.randn_like(x)
            ts_exp = ts.view(-1, 1, 1, 1).expand_as(x)

            lerpd = x * (1. - ts_exp) + ts_exp * eps
            target = eps - x

        pred = self.core(lerpd, z, ts, steps)
        diff_loss = F.mse_loss(pred, target)
        sc_loss = self.get_sc_loss(x_sc,z_sc)

        return diff_loss, sc_loss


if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/diffdec.yml").model
    model = DiffusionDecoderCore(cfg).bfloat16().cuda()
    z = torch.randn(1,128,4,4).bfloat16().cuda()
    with torch.no_grad():
        y = model.sample(z)
        print(y.shape)
    print(cfg)
