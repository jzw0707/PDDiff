from utils import init_env

import argparse
import math

import torch
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader

set_random_seed(42)


def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)

    test_dataset = instantiate_from_config(cfg.test_dataset.mixed)
    

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, test_loader


def build_staged_scheduler(optimizer, cfg):
    warmup_epochs = int(cfg.warmup_epochs)
    unfreeze_epoch = int(cfg.unfreeze_epoch)
    total_epochs = int(cfg.num_epoch)
    head_lr = float(cfg.head_lr)
    backbone_lr = float(cfg.backbone_lr)
    head_min_factor = float(cfg.lr_min) / head_lr
    backbone_min_factor = float(cfg.backbone_lr_min) / backbone_lr

    def cosine_factor(epoch, start_factor, min_factor):
        duration = max(total_epochs - unfreeze_epoch - 1, 1)
        progress = min(max((epoch - unfreeze_epoch) / duration, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (start_factor - min_factor) * cosine

    def head_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        if epoch < unfreeze_epoch:
            return 1.0
        return cosine_factor(epoch, 0.5, head_min_factor)

    def backbone_schedule(epoch):
        if epoch < unfreeze_epoch:
            return 1.0
        return cosine_factor(epoch, 1.0, backbone_min_factor)

    schedules = []
    for group in optimizer.param_groups:
        schedules.append(backbone_schedule if group.get("name") == "backbone" else head_schedule)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results', help='None for saving in wandb folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    cfg = add_args(parser)

    config_pretty(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)
    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)


    train_loader, test_loader = get_loader(cfg)

    if bool(getattr(cfg, "staged_training", False)):
        optimizer_parameters = model.optimizer_parameter_groups(
            head_lr=float(cfg.head_lr),
            backbone_lr=float(cfg.backbone_lr),
        )
    else:
        optimizer_parameters = model.parameters()
    optimizer = instantiate_from_config(cfg.optimizer, params=optimizer_parameters)
    if bool(getattr(cfg, "staged_training", False)):
        scheduler = build_staged_scheduler(optimizer, cfg)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epoch,
            eta_min=cfg.lr_min,
        )

    trainer_class = get_obj_from_str(getattr(cfg, 'trainer', 'utils.trainer.Trainer'))
    trainer = trainer_class(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        fp16=cfg.fp16,
        log_with=getattr(cfg, 'log_with', None),
        cfg=cfg,
    )
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
