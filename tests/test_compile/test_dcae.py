from einops.layers.torch import Reduce
import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..nn.normalization import GroupNorm
from ..nn.resnet import (
    DownBlock, SameBlock, UpBlock,
    LandscapeToSquare, SquareToLandscape
)
from ..nn.sana import ChannelToSpace, SpaceToChannel

from torch.utils.checkpoint import checkpoint

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

class Encoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = nn.Conv2d(3, ch_0, 3, 1, 1, bias = False)
        self.l_to_s = LandscapeToSquare(ch_0) if is_landscape(size) else nn.Sequential()

        blocks = []
        residuals = []
        ch = ch_0

        blocks_per_stage = config.encoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in blocks_per_stage[:-1]:
            next_ch = min(ch*2, ch_max)

            blocks.append(DownBlock(ch, next_ch, block_count, total_blocks))
            residuals.append(SpaceToChannel(ch, next_ch))

            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)

        self.final = SameBlock(ch_max, ch_max, blocks_per_stage[-1], total_blocks)

        self.avg_factor = ch // config.latent_channels
        self.conv_out = nn.Conv2d(ch, config.latent_channels, 1, 1, 0, bias=False)
        self.reduce = Reduce('b (rep c) h w -> b c h w', rep = self.avg_factor, reduction = 'mean')

    def forward(self, x):
        x = self.conv_in(x)
        x = self.l_to_s(x)
        for (block, shortcut) in zip(self.blocks, self.residuals):
            res = shortcut(x)
            x = block(x) + res

        x = self.final(x) + x

        res = x.clone()
        res = self.reduce(res)
        x = self.conv_out(x) + res

        return x

class Decoder(nn.Module):
    def __init__(self, config : 'ResNetConfig', decoder_only = False):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.rep_factor = ch_max // config.latent_channels
        self.conv_in = nn.Conv2d(config.latent_channels, ch_max, 1, 1, 0, bias = False)

        blocks = []
        residuals = []
        ch = ch_0

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        self.starter = SameBlock(ch_max, ch_max, blocks_per_stage[-1], total_blocks)

        for block_count in reversed(blocks_per_stage[:-1]):
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch))

            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))

        self.s_to_l = SquareToLandscape(ch_0) if is_landscape(size) else nn.Sequential()
        self.conv_out = nn.Conv2d(ch_0, 3, 3, 1, 1, bias = False)
        self.norm_out = GroupNorm(ch_0)
        self.act_out = nn.SiLU()

        self.decoder_only = decoder_only
        self.noise_decoder_inputs = config.noise_decoder_inputs


    def forward(self, x):
        if self.decoder_only and self.noise_decoder_inputs > 0.0:
            x = x + torch.randn_like(x) * self.noise_decoder_inputs
            
        res = x.clone()
        res = eo.repeat(res, 'b c h w -> b (rep c) h w', rep = self.rep_factor)

        x = self.conv_in(x) + res
        x = self.starter(x) + x

        for (block, shortcut) in zip(self.blocks, self.residuals):
            res = shortcut(x)
            x = block(x) + res
        
        x = self.s_to_l(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        return x

class DCAE(nn.Module):
    """
    DCAE based autoencoder that takes a ResNetConfig to configure.
    """
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

    def forward(self, x):
        z = self.encoder(x)
        if self.config.noise_decoder_inputs > 0.0:
            dec_input = z + torch.randn_like(z) * self.config.noise_decoder_inputs
        else:
            dec_input = z.clone()

        rec = self.decoder(dec_input)
        return rec, z

def dcae_test():
    from ..configs import ResNetConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ResNetConfig(
        sample_size=256,
        channels=3,
        latent_size=32,
        latent_channels=4,
        noise_decoder_inputs=0.0,
        ch_0=32,
        ch_max=128,
        encoder_blocks_per_stage = [2,2,2,2],
        decoder_blocks_per_stage = [2,2,2,2]
    )

    model = DCAE(cfg).bfloat16().to(device)
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).bfloat16().to(device)
        rec, z, down_rec = model(x)
        assert rec.shape == (1, 3, 256, 256), f"Expected shape (1,3,256,256), got {rec.shape}"
        assert z.shape == (1, 4, 32, 32), f"Expected shape (1,4,32,32), got {z.shape}"
        assert down_rec.shape == (1, 3, 128, 128), f"Expected shape (1,3,128,128), got {down_rec.shape}"
    print("Test passed!")
    
if __name__ == "__main__":
    dcae_test()
