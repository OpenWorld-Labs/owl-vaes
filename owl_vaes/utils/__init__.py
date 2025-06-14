from typing import List
import timeit
import torch
from torch import nn

import time

def freeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = True

class Timer:
    def reset(self):
        self.start_time = time.time()

    def hit(self):
        return time.time() - self.start_time

def versatile_load(path):
    ckpt = torch.load(path, map_location = 'cpu', weights_only=False)
    if 'ema' not in ckpt and 'model' not in ckpt:
        return ckpt
    elif 'ema' in ckpt:
        ckpt = ckpt['ema']
        key_list = list(ckpt.keys())
        ddp_ckpt = False
        for key in key_list:
            if key.startswith("ema_model.module."):
                ddp_ckpt = True
                break
        if ddp_ckpt:
            prefix = 'ema_model.module.'
        else:
            prefix = 'ema_model.'
    elif 'model' in ckpt:
        ckpt = ckpt['model']
        key_list = list(ckpt.keys())
        ddp_ckpt = False
        for key in key_list:
            if key.startswith("module."):
                ddp_ckpt = True
        if ddp_ckpt:
            prefix = 'module.'
        else:
            prefix = None

    if prefix is None:
        return ckpt
    else:
        ckpt = {k[len(prefix):] : v for (k,v) in ckpt.items() if k.startswith(prefix)}

    return ckpt

def prefix_filter(ckpt, prefix):
    return {k[len(prefix):] : v for (k,v) in ckpt.items() if k.startswith(prefix)}

def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end - begin), torch.cuda.max_memory_allocated()

def custom_viz_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    From torch.compile docs, for visualizing the FX graph of a model.
    It takes the inputs but doesn't run the model using them.
    Also, the optimized kernels are not captured, only the FX graph is printed.
    Each backend will optimize the graph differently based on their kernels.
    """
    print("Custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward
