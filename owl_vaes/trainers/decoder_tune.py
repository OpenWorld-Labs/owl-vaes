"""
Trainer for reconstruction only
"""

from einops.layers.torch import Reduce
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..discriminators import get_discriminator_cls
from ..muon import init_muon
from ..nn.lpips import get_lpips_cls
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb
from .base import BaseTrainer
from ..configs import Config


def latent_reg_loss(z):
    # z is [b,c,h,w]
    loss = z.pow(2)
    loss = Reduce('b c h w -> b', reduction = 'sum')(loss).mean()
    return 0.5 * loss

class DecTuneTrainer(BaseTrainer):
    """
    Trainer for only the decoder, with frozen encoder.
    Does L2 + LPIPS + GAN

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg_path = self.train_cfg.teacher_cfg

        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model

        teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        teacher.load_state_dict(teacher_ckpt)

        self.encoder = teacher.encoder

        if self.train_cfg.use_teacher_decoder:
            self.model = teacher.decoder
            self.model.decoder_only = True
        else:
            del teacher.decoder
            model_id = self.model_cfg.model_id
            model = get_model_cls(model_id)(self.model_cfg)
            del model.encoder
            self.model = model.decoder

        disc_cfg = self.model_cfg.discriminator
        self.discriminator = get_discriminator_cls(disc_cfg.model_id)(disc_cfg)

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"Model parameters: {model_params:,}")
            print(f"Discriminator parameters: {disc_params:,}")

        self.ema = None
        self.opt = None
        self.d_opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.model.load_state_dict(save_dict['model'])
        self.discriminator.load_state_dict(save_dict['discriminator'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        # Loss weights
        lpips_weight = self.train_cfg.loss_weights.get('lpips', 0.0)
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)
        r12_weight = self.train_cfg.loss_weights.get('r12', 0.0)

        # Prepare model, lpips, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model)
        
        self.discriminator = self.discriminator.to(self.device).train()
        if self.world_size > 1:
            self.discriminator = DDP(self.discriminator)
        freeze(self.discriminator)

        self.lpips = None
        if lpips_weight > 0.0:
            lpips = get_lpips_cls(self.train_cfg.lpips_id)(self.device).to(self.device).eval()
            freeze(lpips)

        self.encoder = self.encoder.to(self.device).bfloat16().eval()
        freeze(self.encoder)

        self.encoder = torch.compile(self.encoder)
        #self.lpips.model = torch.compile(self.lpips.model)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # compile all training and frozen models
        self.encoder = torch.compile(self.encoder, mode="max-autotune", fullgraph=True)
        self.lpips = torch.compile(self.lpips, mode="max-autotune", fullgraph=True)
        self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
        self.discriminator = torch.compile(self.discriminator, mode="max-autotune", fullgraph=True)
        self.ema = torch.compile(self.ema, mode="max-autotune", fullgraph=True)

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
            self.d_opt = init_muon(self.discriminator, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)
            self.d_opt = opt_cls(self.discriminator.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size)

        def warmup_gan_weight():
            if self.total_step_counter < self.train_cfg.delay_adv:
                return 0.0
            else:
                x = (self.total_step_counter - self.train_cfg.delay_adv) / self.train_cfg.warmup_adv
                x = max(0.0, min(1.0, x))
                # Cosine annealing from 0 to 1
                ramp = 0.5 * (1 - torch.cos(torch.tensor(x * torch.pi)).item())
                return ramp * gan_weight  

        def r_loss(d, x, sigma = 0.01):
            z = sigma * torch.randn_like(x)
            d_clean = d(x.detach())
            d_noisy = d(x.detach() + z)

            return ((d_clean - d_noisy).pow(2).mean())
        
        def d_loss(d, x_fake, x_real):
            fake_out = d(x_fake)
            real_out = d(x_real)

            fake_loss = F.relu(1+fake_out).mean()
            real_loss = F.relu(1-real_out).mean()

            return fake_loss + real_loss

        def g_loss(d, x_fake):
            return -d(x_fake).mean()

        def merged_d_losses(d, x_fake, x_real, sigma=0.01):
            fake_out = d(x_fake.detach())
            real_out = d(x_real.detach())

            fake_out_noisy = d((x_fake + sigma*torch.randn_like(x_fake)).detach())
            real_out_noisy = d((x_real + sigma*torch.randn_like(x_real)).detach())

            r1_penalty = (fake_out_noisy - fake_out).pow(2).mean()
            r2_penalty = (real_out_noisy - real_out).pow(2).mean()

            fake_loss = F.relu(1 + fake_out).mean()
            real_loss = F.relu(1 - real_out).mean()

            return r1_penalty, r2_penalty, (fake_loss + real_loss)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()

                with torch.no_grad() and ctx:
                    teacher_z = self.encoder(batch) / self.train_cfg.latent_scale
                    teacher_z = F.interpolate(teacher_z, scale_factor=.5)
                    batch = F.interpolate(batch, scale_factor=.5)
                with ctx:
                    batch_rec = self.model(teacher_z)

                # Discriminator training
                unfreeze(self.discriminator)
                with ctx:
                    if r12_weight == 0.0:
                        disc_loss = d_loss(self.discriminator, batch_rec.detach(), batch.detach()) / accum_steps
                        metrics.log('disc_loss', disc_loss)
                    else:
                        r1_penalty, r2_penalty, d_loss = merged_d_losses(
                            self.discriminator, 
                            batch_rec.detach(), 
                            batch.detach()
                        )
                        r1_penalty = r1_penalty / accum_steps
                        r2_penalty = r2_penalty / accum_steps
                        d_loss = d_loss / accum_steps

                        metrics.log('r1_penalty', r1_penalty)
                        metrics.log('r2_penalty', r2_penalty)
                        metrics.log('disc_loss', d_loss)

                        disc_loss = d_loss + (r1_penalty + r2_penalty) * 0.5 * r12_weight


                self.scaler.scale(disc_loss).backward()
                freeze(self.discriminator)

                with ctx:
                    mse_loss = F.mse_loss(batch_rec, batch) / accum_steps
                    total_loss += mse_loss
                metrics.log('mse_loss', mse_loss)

                if lpips_weight > 0.0:
                    with ctx:
                        lpips_loss = self.lpips(batch_rec, batch) / accum_steps
                        total_loss += lpips_loss
                    metrics.log('lpips_loss', lpips_loss)

                crnt_gan_weight = warmup_gan_weight()
                if crnt_gan_weight > 0.0:
                    with ctx:
                        gan_loss = g_loss(self.discriminator, batch_rec) / accum_steps
                        total_loss += crnt_gan_weight * gan_loss
                metrics.log('gan_loss', gan_loss)

                self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.unscale_(self.d_opt)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.step(self.d_opt)
                    self.d_opt.zero_grad(set_to_none=True)

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    with ctx:
                        self.ema.update()

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                ema_rec = self.ema.ema_model(teacher_z)

                            wandb_dict['samples'] = to_wandb(
                                batch.detach().contiguous().bfloat16(),
                                ema_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
