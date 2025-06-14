"""
Base class for any trainer
"""

import os

import torch
import torch.distributed as dist
import wandb


class BaseTrainer:
    def __init__(
        self,
        train_cfg, logging_cfg, model_cfg,
        global_rank = 0, local_rank = 0, world_size = 1,
        device = None
    ):
        self.rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.train_cfg = train_cfg
        self.logging_cfg = logging_cfg
        self.model_cfg = model_cfg

        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{self.local_rank}')
            else:
                device = torch.device('cpu')
        elif type(device) == str:
            if device.find(':') != -1:
                # device is a string of the form "cuda:0"
                device = torch.device(device)
            else:
                device = torch.device(device, self.local_rank)
        self.device = device

        if 'cuda' in self.device.type:
            torch.cuda.set_device(self.device)

        if self.logging_cfg is not None and self.rank == 0:
            log = self.logging_cfg
            wandb.init(
                project=log.project,
                entity=log.name,
                name=log.run_name,
                config={"train": train_cfg, "model": model_cfg},
            )

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def get_module(self, ema: bool = False):
        if self.world_size == 1:
            if ema:
                return self.ema.ema_model
            else:
                return self.model
        else:
            if ema:
                return self.ema.ema_model.module
            else:
                return self.model.module

    def save(self, save_dict):
        os.makedirs(self.train_cfg.checkpoint_dir, exist_ok=True)

        fp = os.path.join(
            self.train_cfg.checkpoint_dir, f"step_{self.total_step_counter}.pt"
        )

        torch.save(save_dict, fp)

    def load(self, path: str):
        return torch.load(path, map_location="cpu", weights_only=False)
