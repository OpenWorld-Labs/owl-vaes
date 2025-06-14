import torch
import torch.nn.functional as F
from torch import nn
import math
import einops as eo

from ..nn.audio_blocks import ResBlock, SnakeBeta

from torch.utils.checkpoint import checkpoint

class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride, total_blocks):
        super().__init__()

        self.block1 = ResBlock(ch_in, 1, total_blocks * 3)
        self.block2 = ResBlock(ch_in, 3, total_blocks * 3)
        self.block3 = ResBlock(ch_in, 9, total_blocks * 3)
    
        self.act = SnakeBeta(ch_in)
        self.proj = nn.Conv1d(ch_in, ch_out, 2*stride,stride,math.ceil(stride/2), bias = False)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.act(x)
        x = self.proj(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride, total_blocks):
        super().__init__()

        self.block1 = ResBlock(ch_out, 1, total_blocks * 3)
        self.block2 = ResBlock(ch_out, 3, total_blocks * 3)
        self.block3 = ResBlock(ch_out, 9, total_blocks * 3)

        self.act = SnakeBeta(ch_out)

        self.proj = nn.Conv1d(ch_in, ch_out, 2*stride, stride=1, bias = False, padding = 'same')
        self.stride = stride

    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.stride, mode = 'nearest')
        x = self.proj(x)

        x = self.act(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride, total_blocks):
        super().__init__()

        self.block1 = ResBlock(ch_in, 1, total_blocks * 3)
        self.block2 = ResBlock(ch_in, 3, total_blocks * 3)
        self.block3 = ResBlock(ch_in, 9, total_blocks * 3)
    
        self.act = SnakeBeta(ch_in)
        self.proj = nn.Conv1d(ch_in, ch_out, 1,1,0, bias = False)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.act(x)
        x = self.proj(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = nn.Conv1d(config.channels, ch_0, 7, 1, 3, bias=False)

        blocks = []
        ch = ch_0

        strides = config.strides
        total_blocks = len(strides)

        for stride in strides[:-1]:
            next_ch = min(ch*2, ch_max)
            blocks.append(EncoderBlock(ch, next_ch, stride, total_blocks))
            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.final = Block(ch, ch, strides[-1], total_blocks)
        self.conv_out = nn.Conv1d(ch, config.latent_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.final(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = nn.Conv1d(config.latent_channels, ch_max, 1, 1, 0, bias=False)
        
        
        blocks = []

        strides = config.strides
        total_blocks = len(strides)

        self.starter = Block(ch_max, ch_max, strides[-1], total_blocks)

        ch = ch_0
        for stride in strides[:-1]:
            next_ch = min(ch*2, ch_max)
            blocks.append(DecoderBlock(next_ch, ch, stride, total_blocks))
            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))


        self.conv_out = nn.Conv1d(ch_0, config.channels, 7, 1, 3, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.starter(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)
        return torch.tanh(x)

class OobleckVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.eq = config.eq
        self.checkpoint_grads = config.checkpoint_grads
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        if self.eq:
            n = z.shape[-1]
            n = n // 3
            z_1 = z[:,:,:2*n] # First 2/3
            z_2 = z[:,:,n:] # Last 2/3
            rec_1 = self.decode(z_1)
            rec_2 = self.decode(z_2)
            return x_rec, z, (rec_1, rec_2)
        else:
            return x_rec, z
