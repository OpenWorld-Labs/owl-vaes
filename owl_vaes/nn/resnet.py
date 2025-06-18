from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .normalization import RMSNorm2d, GroupNorm
from torch.nn.utils.parametrizations import weight_norm

"""
Building blocks for any ResNet based model
"""

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)

class ResBlock(nn.Module):
    """
    Basic ResNet block from R3GAN paper using

    :param ch: Channel count to use for the block
    :param total_res_blocks: How many res blocks are there in the entire model?
    """
    def __init__(self, ch, total_res_blocks):
        super().__init__()

        grp_size = 16
        n_grps = (2*ch) // grp_size

        self.conv1 = weight_norm(nn.Conv2d(ch, 2*ch, 1, 1, 0))
        #self.norm1 = RMSNorm2d(2*ch)
        #self.norm1 = GroupNorm(2*ch, n_grps)

        self.conv2 = weight_norm(nn.Conv2d(2*ch, 2*ch, 3, 1, 1, groups = n_grps))
        #self.norm2 = RMSNorm2d(2*ch)
        #self.norm2 = GroupNorm(2*ch, n_grps)

        self.conv3 = weight_norm(nn.Conv2d(2*ch, ch, 1, 1, 0, bias=False))

        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        # Fix up init
        scaling_factor = total_res_blocks ** -.25

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv1.weight.data *= scaling_factor

        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.conv2.weight.data *= scaling_factor

        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        res = x.clone()

        def _inner(x):
            x = self.conv1(x)
            #x = self.norm1(x)
            x = self.act1(x)
            x = self.conv2(x)
            #x = self.norm2(x)
            x = self.act2(x)
            x = self.conv3(x)
            return x

        if self.training:
            x = checkpoint(_inner, x)
        else:
            x = _inner(x)

        return x + res

class Upsample(nn.Module):
    """
    Bilinear upsample + project layer
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = nn.Sequential() if ch_in == ch_out else weight_norm(nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False))

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, scale_factor = 2,  mode = 'bicubic')
        return x

class Downsample(nn.Module):
    """
    Bilinear downsample + project layer
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = weight_norm(nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False))

    def forward(self, x):
        x = F.interpolate(x, scale_factor = .5, mode = 'bicubic')
        x = self.proj(x)
        return x

class UpBlock(nn.Module):
    """
    General upsampling stage block
    """
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        self.up = Upsample(ch_in, ch_out)
        blocks = []
        num_total = num_res * total_blocks
        for _ in range(num_res):
            blocks.append(ResBlock(ch_out, num_total))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.up(x)
        for block in self.blocks:
            x = block(x)
        return x

class DownBlock(nn.Module):
    """
    General downsampling stage block
    """
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        self.down = Downsample(ch_in, ch_out)
        blocks = []
        num_total = num_res * total_blocks
        for _ in range(num_res):
            blocks.append(ResBlock(ch_in, num_total))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.down(x)
        return x

class SameBlock(nn.Module):
    """
    General block with no up/down
    """
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        blocks = []
        num_total = num_res * total_blocks
        for _ in range(num_res):
            blocks.append(ResBlock(ch_in, num_total))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class LandscapeToSquare(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.proj = weight_norm(nn.Conv2d(ch, ch, 3, 1, 1, bias = False))
    
    def forward(self, x):
        # x is [9, 16]
        _,_,h,w = x.shape

        h_mult = (512/360)
        w_mult = (512/640)

        new_h = round(h * h_mult)
        new_w = round(w * w_mult)

        x = F.interpolate(x, (new_h, new_w), mode='bicubic')
        x = self.proj(x)
        return x

class SquareToLandscape(LandscapeToSquare):
    def forward(self, x):
        # x is [1,1]
        _,_,h,w = x.shape

        h_mult = (360/512)
        w_mult = (640/512)

        new_h = round(h*h_mult)
        new_w = round(w*w_mult)

        x = self.proj(x)
        x = F.interpolate(x, (new_h, new_w), mode='bicubic')
        return x
