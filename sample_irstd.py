import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from utils import init_env
from utils.collate_utils import collate
from utils.import_utils import get_obj_from_str, instantiate_from_config, recurse_instantiate_from_config
from utils.init_utils import add_args
from utils.irstd_trainer import evaluate_irstd
from utils.train_utils import set_random_seed


set_random_seed(7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results_irstd")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_sample_steps", type=int, default=None)
    parser.add_argument("--target_dataset", nargs="+", default=["SIRST4", "SIRST-UAV"])
    parser.add_argument("--fp16", action="store_true")
    cfg = add_args(parser)

    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps
    missing = [name for name in cfg.target_dataset if name not in cfg.test_dataset]
    if missing:
        raise ValueError(f"Unknown test datasets: {missing}; available: {list(cfg.test_dataset.keys())}")

    cond_uvit = instantiate_from_config(
        cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass),
    )
    denoiser = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
    model = instantiate_from_config(cfg.diffusion_model, model=denoiser)

    checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"Checkpoint load: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

    accelerator = Accelerator(mixed_precision="fp16" if cfg.fp16 else "no")
    model = accelerator.prepare(model)
    forward_fn = get_obj_from_str(cfg.train_val_forward_fn)
    all_metrics = {}

    for dataset_name in cfg.target_dataset:
        dataset = instantiate_from_config(cfg.test_dataset[dataset_name])
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate,
        )
        loader = accelerator.prepare(loader)
        save_to = Path(cfg.results_folder) / dataset_name
        metrics = evaluate_irstd(model, loader, accelerator, forward_fn, save_to=save_to)
        all_metrics[dataset_name] = metrics
        accelerator.print(dataset_name, json.dumps(metrics, indent=2))

    if accelerator.is_main_process:
        results_path = Path(cfg.results_folder) / "metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

