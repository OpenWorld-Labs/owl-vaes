import argparse
import os
import torch

from owl_vaes.configs import Config
from owl_vaes.trainers import get_trainer_cls
from owl_vaes.utils.ddp import cleanup, setup

if __name__ == "__main__":
    # torch compile flag to convert conv with 1x1 kernel to matrix multiplication
    torch._inductor.config.conv_1x1_as_mm = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.benchmark = True


    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)

    global_rank, local_rank, world_size = setup()

    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train, cfg.wandb, cfg.model, global_rank, local_rank, world_size
    )

    trainer.train()
    cleanup()
