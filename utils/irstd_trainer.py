import copy
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.collate_utils import collate
from utils.import_utils import fill_args_from_dict
from utils.import_utils import instantiate_from_config
from utils.metrics import IRSTDMetricAccumulator
from utils.train_utils import SmoothedValue
from utils.trainer import Trainer


def restore_prediction(prediction, original_size, resized_size, padding):
    original_h, original_w = (int(value) for value in original_size)
    resized_h, resized_w = (int(value) for value in resized_size)
    top, left = (int(value) for value in padding)
    prediction = prediction[..., top : top + resized_h, left : left + resized_w]
    prediction = F.interpolate(
        prediction.unsqueeze(0),
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )
    return prediction.squeeze().clamp(0, 1).cpu().numpy()


@torch.inference_mode()
def evaluate_irstd(model, data_loader, accelerator, forward_fn, save_to=None, threshold=0.5):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    accumulator = IRSTDMetricAccumulator(threshold=threshold)
    save_path = Path(save_to) if save_to is not None else None
    if save_path is not None and accelerator.is_main_process:
        save_path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    for data in tqdm(data_loader, disable=not accelerator.is_main_process):
        image = data["image"].to(accelerator.device)
        output = forward_fn(unwrapped_model, image=image, verbose=False)
        predictions = output["pred"].detach().cpu()
        batch_values = []
        for index, prediction in enumerate(predictions):
            restored = restore_prediction(
                prediction,
                data["original_size"][index],
                data["resized_size"][index],
                data["padding"][index],
            )
            target = np.asarray(data["gt"][index], dtype=np.uint8) > 0
            sample_metrics = IRSTDMetricAccumulator(threshold=threshold)
            sample_metrics.update(restored, target)
            batch_values.append(sample_metrics.values)
            if save_path is not None:
                Image.fromarray(np.round(restored * 255).astype(np.uint8)).save(save_path / data["name"][index])

        # Accelerate pads the final distributed batch with repeated samples.
        # Gather per-image statistics so those repeats are trimmed before summing.
        raw = torch.from_numpy(np.stack(batch_values)).to(accelerator.device)
        gathered = accelerator.gather_for_metrics(raw)
        accumulator.values += gathered.sum(dim=0).cpu().numpy()

    return IRSTDMetricAccumulator.compute(accumulator.values)


class IRSTDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("cfg")
        self.cfg = cfg
        self.val_interval = int(getattr(cfg, "val_interval", 5))
        # DDP only registers gradient hooks for parameters trainable at wrapping
        # time. Register the later-unfrozen stages too, then restore the warmup
        # mask once DDP has been constructed.
        initial_trainable = None
        model = args[0] if args else kwargs["model"]
        if bool(getattr(cfg, "staged_training", False)):
            initial_trainable = [parameter.requires_grad for parameter in model.parameters()]
            model.model.set_trainable_backbone_stages(
                int(getattr(cfg, "unfreeze_last_stages", 2)),
                stage_indices=list(getattr(cfg, "unfreeze_stage_indices", [])),
                unfreeze_patch_embed=bool(getattr(cfg, "unfreeze_patch_embed", False)),
            )
        super().__init__(*args, **kwargs)
        if initial_trainable is not None:
            for parameter, trainable in zip(model.parameters(), initial_trainable):
                parameter.requires_grad_(trainable)
        self.best_iou = -1.0
        self.best_score = -1.0
        self.unfreeze_epoch = int(getattr(cfg, "unfreeze_epoch", 20))
        self.unfreeze_last_stages = int(getattr(cfg, "unfreeze_last_stages", 2))
        self.unfreeze_stage_indices = list(getattr(cfg, "unfreeze_stage_indices", []))
        self.unfreeze_patch_embed = bool(getattr(cfg, "unfreeze_patch_embed", False))
        self.noise_curriculum_start = float(getattr(cfg, "noise_curriculum_start", 1.0))
        self.ema_decay = float(getattr(cfg, "ema_decay", 0.0))
        self._backbone_unfrozen = False
        self.ema_model = None
        if self.ema_decay > 0:
            self.ema_model = copy.deepcopy(self.accelerator.unwrap_model(self.model))
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

        self.evaluation_loaders = {}
        evaluation_datasets = list(getattr(cfg, "evaluation_datasets", []))
        for dataset_name in evaluation_datasets:
            dataset = instantiate_from_config(cfg.test_dataset[dataset_name])
            loader = DataLoader(
                dataset,
                batch_size=int(getattr(cfg, "eval_batch_size", cfg.batch_size)),
                num_workers=cfg.num_workers,
                collate_fn=collate,
            )
            self.evaluation_loaders[dataset_name] = self.accelerator.prepare(loader)

    def _diffusion_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _configure_epoch(self, epoch):
        diffusion_model = self._diffusion_model()
        if self.unfreeze_epoch > 0:
            progress = min((epoch + 1) / self.unfreeze_epoch, 1.0)
            diffusion_model.train_time_max = self.noise_curriculum_start + (
                1.0 - self.noise_curriculum_start
            ) * progress
        else:
            diffusion_model.train_time_max = 1.0

        if epoch >= self.unfreeze_epoch and not self._backbone_unfrozen:
            diffusion_model.model.set_trainable_backbone_stages(
                self.unfreeze_last_stages,
                stage_indices=self.unfreeze_stage_indices,
                unfreeze_patch_embed=self.unfreeze_patch_embed,
            )
            self._backbone_unfrozen = True
            trainable = sum(parameter.numel() for parameter in diffusion_model.parameters() if parameter.requires_grad)
            self.logger.info(
                f"Epoch:{epoch}/{self.train_num_epoch} unfroze SAM2 stages "
                f"last={self.unfreeze_last_stages} selected={self.unfreeze_stage_indices} "
                f"patch_embed={self.unfreeze_patch_embed}; trainable_params:{trainable}"
            )

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        training_model = self._diffusion_model()
        update_weight = 1.0 - self.ema_decay
        for ema_parameter, parameter in zip(self.ema_model.parameters(), training_model.parameters()):
            ema_parameter.lerp_(parameter.detach(), update_weight)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), training_model.buffers()):
            ema_buffer.copy_(buffer.detach())

    def _validation_metrics(self, model, accelerator):
        if not self.evaluation_loaders:
            return {"mixed": self.val(model, self.test_loader, accelerator)}
        return {
            dataset_name: self.val(model, loader, accelerator)
            for dataset_name, loader in self.evaluation_loaders.items()
        }

    def val(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        return evaluate_irstd(
            model,
            test_data_loader,
            accelerator,
            self.train_val_forward_fn,
            save_to=save_to,
        )

    def train(self):
        accelerator = self.accelerator
        self.logger.info(
            f"Training processes:{accelerator.num_processes} "
            f"local_batch_size:{self.train_loader.batch_sampler.batch_size} "
            f"steps_per_epoch:{len(self.train_loader)} epochs:{self.train_num_epoch}"
        )
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            self._configure_epoch(epoch)
            self.model.train()
            loss_sm = SmoothedValue(window_size=10)
            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as progress:
                for data in self.train_loader:
                    with accelerator.autocast(), accelerator.accumulate(self.model):
                        loss = fill_args_from_dict(self.train_val_forward_fn, data)(model=self.model)
                        accelerator.backward(loss)
                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        if accelerator.sync_gradients:
                            self._update_ema()
                        self.opt.zero_grad()
                    loss_sm.update(loss.item())
                    progress.set_description(
                        f"Epoch:{epoch}/{self.train_num_epoch} loss:{loss_sm.avg:.4f}({loss_sm.global_avg:.4f})"
                    )
                    learning_rates = {
                        f"train/lr_{group.get('name', index)}": group["lr"]
                        for index, group in enumerate(self.opt.param_groups)
                    }
                    accelerator.log({"train/loss": loss_sm.avg, **learning_rates})
                    progress.update()
                    log_interval = int(getattr(self.cfg, "log_interval", 0))
                    if log_interval and progress.n % log_interval == 0:
                        self.logger.info(
                            f"Epoch:{epoch}/{self.train_num_epoch} "
                            f"step:{progress.n}/{len(self.train_loader)} "
                            f"loss:{loss_sm.global_avg:.6f}"
                        )

            if self.scheduler is not None:
                self.scheduler.step()
            accelerator.wait_for_everyone()
            mean_loss = accelerator.gather(
                torch.tensor([loss_sm.global_avg], device=accelerator.device)
            ).mean().item()
            self.logger.info(f"Epoch:{epoch}/{self.train_num_epoch} loss:{mean_loss:.4f}")

            should_validate = epoch == 0 or (epoch + 1) % self.val_interval == 0 or epoch + 1 == self.train_num_epoch
            if should_validate:
                validation_model = self.ema_model if self.ema_model is not None else self.model
                dataset_metrics = self._validation_metrics(validation_model, accelerator)
                for dataset_name, metrics in dataset_metrics.items():
                    metrics_text = " ".join(f"{key}:{value:.6f}" for key, value in metrics.items())
                    self.logger.info(
                        f"Epoch:{epoch}/{self.train_num_epoch} test/{dataset_name} {metrics_text}"
                    )
                    accelerator.log(
                        {f"test/{dataset_name}/{key}": value for key, value in metrics.items()}
                    )
                mean_iou = float(np.mean([metrics["IoU"] for metrics in dataset_metrics.values()]))
                score_weights = getattr(self.cfg, "dataset_score_weights", None)
                if score_weights:
                    weighted_scores = [
                        (float(score_weights[name]), metrics["IoU"])
                        for name, metrics in dataset_metrics.items()
                        if name in score_weights
                    ]
                    weight_sum = sum(weight for weight, _ in weighted_scores)
                    score = float(
                        sum(weight * value for weight, value in weighted_scores) / weight_sum
                    )
                else:
                    score = float(np.mean([metrics["IoU"] for metrics in dataset_metrics.values()]))
                self.logger.info(
                    f"Epoch:{epoch}/{self.train_num_epoch} validation selection_score:{score:.6f}"
                )
                if accelerator.is_main_process:
                    record = {
                        "epoch": epoch,
                        "loss": mean_loss,
                        "learning_rates": {
                            group.get("name", str(index)): group["lr"]
                            for index, group in enumerate(self.opt.param_groups)
                        },
                        "mean_IoU": mean_iou,
                        "selection_score": score,
                        "datasets": dataset_metrics,
                    }
                    with (self.results_folder / "metrics.jsonl").open("a") as metrics_file:
                        metrics_file.write(json.dumps(record) + "\n")
                if score > self.best_score:
                    self.best_score = score
                    self.best_iou = score
                    model_state = (
                        self.ema_model.state_dict()
                        if self.ema_model is not None
                        else accelerator.get_state_dict(self.model)
                    )
                    self.save(
                        "best",
                        keep_previous=False,
                        model_state=model_state,
                        extra_state={
                            "best_iou": self.best_iou,
                            "best_score": self.best_score,
                            "best_metrics": dataset_metrics,
                            "ema_decay": self.ema_decay,
                        },
                    )
                    self.logger.info(
                        f"Epoch:{epoch}/{self.train_num_epoch} saved model-best.pt "
                        f"(selection_score:{self.best_score:.6f})"
                    )
                    if accelerator.is_main_process:
                        (self.results_folder / "best_metrics.json").write_text(
                            json.dumps(record, indent=2) + "\n"
                        )
            accelerator.wait_for_everyone()

        self.logger.info("training complete")
        accelerator.end_training()
