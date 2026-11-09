import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from model.dmsp import DMSP, apply_dmsp
from model.msa import MSA, apply_msa
from model.net import Decoder, SAM2backbone, timestep_embedding


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )


class ResidualRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.SiLU()

    def forward(self, feature):
        return self.activation(feature + self.block(feature))


class DynamicSpectralContext(nn.Module):
    """Resolution-independent frequency context for the deepest FPN feature."""

    def __init__(self, channels):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 1, groups=2 * channels),
            nn.SiLU(),
            nn.Conv2d(2 * channels, 2 * channels, 1),
        )

    def forward(self, feature):
        dtype = feature.dtype
        height, width = feature.shape[-2:]
        spectrum = torch.fft.rfft2(feature.float(), norm="ortho")
        filtered = self.filter(torch.cat([spectrum.real, spectrum.imag], dim=1))
        real, imaginary = filtered.chunk(2, dim=1)
        context = torch.fft.irfft2(
            torch.complex(real, imaginary),
            s=(height, width),
            norm="ortho",
        )
        return context.to(dtype)


class DetailFusion(nn.Module):
    """Fuse semantic features with the infrared detail branch."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.semantic_qk = nn.Conv2d(channels, 2 * channels, 1)
        self.detail_qk = nn.Conv2d(channels, 2 * channels, 1)
        self.value = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.mutual_norm = nn.GroupNorm(min(8, channels), channels)
        self.difference_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.branch_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1),
        )
        self.output = nn.Conv2d(channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, 2 * channels),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, semantic, detail, timesteps):
        semantic_query, semantic_key = self.semantic_qk(semantic).chunk(2, dim=1)
        detail_query, detail_key = self.detail_qk(detail).chunk(2, dim=1)
        value_mutual, value_difference = self.value(
            torch.cat([semantic, detail], dim=1)
        ).chunk(2, dim=1)

        semantic_spectrum = torch.fft.rfft2(semantic_query.float(), norm="ortho")
        detail_spectrum = torch.fft.rfft2(detail_key.float(), norm="ortho")
        mutual = torch.fft.irfft2(
            semantic_spectrum * detail_spectrum.conj(),
            s=semantic.shape[-2:],
            norm="ortho",
        ).to(semantic.dtype)
        mutual = self.mutual_norm(mutual) * value_mutual

        difference = self.difference_gate(
            torch.abs(semantic_key - detail_query)
        ) * value_difference
        branch_weights = self.branch_gate(torch.cat([mutual, difference], dim=1))
        fused = semantic + self.output(
            branch_weights[:, :1] * mutual + branch_weights[:, 1:] * difference
        )

        time_scale, time_shift = self.time_mlp(
            timestep_embedding(timesteps, self.channels)
        ).chunk(2, dim=1)
        return fused * (1 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]


class HighResolutionIRSTDDecoder(nn.Module):
    """Top-down semantic FPN with full-resolution image/mask detail skips."""

    def __init__(
        self,
        feature_dims,
        decoder_dim=96,
        class_num=1,
        use_detail_fusion=False,
        targetness_head=False,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.use_detail_fusion = use_detail_fusion
        self.lateral = nn.ModuleList(
            nn.Conv2d(feature_dim, decoder_dim, 1) for feature_dim in feature_dims
        )
        self.spectral_context = DynamicSpectralContext(decoder_dim)
        self.fpn_refine = nn.ModuleList(ResidualRefine(decoder_dim) for _ in range(3))
        self.dmsp = nn.ModuleList(DMSP(decoder_dim) for _ in range(4))
        self.msa = nn.ModuleList(
            MSA(
                decoder_dim,
                in_size=patch,
                patch=patch,
                mem_slots=slots,
                topk=topk,
            )
            for patch, slots, topk in ((16, 256, 8), (16, 128, 8), (8, 64, 8), (4, 32, 4))
        )

        detail_dim = decoder_dim // 3
        half_dim = decoder_dim // 2
        self.detail_full = ConvGNAct(4, detail_dim)
        self.detail_half = nn.Sequential(
            nn.Conv2d(detail_dim, half_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, half_dim), half_dim),
            nn.SiLU(),
        )
        self.detail_quarter = nn.Sequential(
            nn.Conv2d(half_dim, decoder_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, decoder_dim), decoder_dim),
            nn.SiLU(),
        )
        if use_detail_fusion:
            self.contrast_projection = nn.Conv2d(1, detail_dim, 3, padding=1, bias=False)
            self.detail_eighth = nn.Sequential(
                nn.Conv2d(decoder_dim, decoder_dim, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(8, decoder_dim), decoder_dim),
                nn.SiLU(),
            )
            self.semantic_detail_fusion = nn.ModuleList(
                DetailFusion(decoder_dim) for _ in range(2)
            )
            nn.init.zeros_(self.contrast_projection.weight)

        self.quarter_fuse = nn.Sequential(
            ConvGNAct(2 * decoder_dim, decoder_dim),
            ResidualRefine(decoder_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(decoder_dim, 2 * decoder_dim),
            nn.SiLU(),
            nn.Linear(2 * decoder_dim, 2 * decoder_dim),
        )
        self.up_half = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, half_dim, 4, stride=2, padding=1),
            ConvGNAct(half_dim, half_dim),
        )
        self.half_fuse = nn.Sequential(
            ConvGNAct(2 * half_dim, half_dim),
            ResidualRefine(half_dim),
        )
        self.up_full = nn.Sequential(
            nn.ConvTranspose2d(half_dim, detail_dim, 4, stride=2, padding=1),
            ConvGNAct(detail_dim, detail_dim),
        )
        self.full_fuse = nn.Sequential(
            ConvGNAct(2 * detail_dim, detail_dim),
            ResidualRefine(detail_dim),
        )
        self.prediction = nn.Conv2d(detail_dim, class_num, 1)
        self.targetness_prediction = (
            nn.Conv2d(detail_dim, class_num, 1) if targetness_head else None
        )
        self.auxiliary_heads = nn.ModuleList(
            nn.Conv2d(decoder_dim, class_num, 1) for _ in range(3)
        )

    def _align(self, index, feature):
        feature = apply_dmsp(self.dmsp[index], feature)
        return apply_msa(self.msa[index], feature)

    def forward(self, features, timesteps, noisy_mask, image):
        c1, c2, c3, c4 = features
        p4 = self._align(3, self.lateral[3](c4))
        p4 = p4 + self.spectral_context(p4)
        p3 = self.fpn_refine[2](
            self.lateral[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        )
        p3 = self._align(2, p3)

        detail_full = self.detail_full(torch.cat([image, noisy_mask], dim=1))
        if self.use_detail_fusion:
            infrared = image.mean(dim=1, keepdim=True)
            local_contrast = infrared - F.avg_pool2d(infrared, 7, stride=1, padding=3)
            detail_full = detail_full + self.contrast_projection(local_contrast)
        detail_half = self.detail_half(detail_full)
        detail_quarter = self.detail_quarter(detail_half)

        p2 = self.fpn_refine[1](
            self.lateral[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        )
        p2 = self._align(1, p2)
        if self.use_detail_fusion:
            detail_eighth = self.detail_eighth(detail_quarter)
            p2 = self.semantic_detail_fusion[1](p2, detail_eighth, timesteps)
        p1 = self.fpn_refine[0](
            self.lateral[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        )
        p1 = self._align(0, p1)
        if self.use_detail_fusion:
            p1 = self.semantic_detail_fusion[0](p1, detail_quarter, timesteps)
        quarter = self.quarter_fuse(torch.cat([p1, detail_quarter], dim=1))

        time_scale, time_shift = self.time_mlp(
            timestep_embedding(timesteps, self.decoder_dim)
        ).chunk(2, dim=1)
        quarter = quarter * (1 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]

        half = self.up_half(quarter)
        half = self.half_fuse(torch.cat([half, detail_half], dim=1))
        full = self.up_full(half)
        full = self.full_fuse(torch.cat([full, detail_full], dim=1))
        prediction = self.prediction(full)

        if not self.training:
            return prediction
        output_size = prediction.shape[-2:]
        auxiliary = [
            F.interpolate(head(feature), size=output_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.auxiliary_heads, (p1, p2, p3))
        ]
        if self.targetness_prediction is not None:
            return {
                "prediction": prediction,
                "auxiliary_outputs": tuple(auxiliary),
                "targetness": self.targetness_prediction(full),
            }
        return (prediction, *auxiliary)


class PDDiff(nn.Module):
    """PDDiff infrared denoiser."""

    BACKBONES = {
        "tiny": ("sam2_hiera_t.yaml", [96, 192, 384, 768]),
        "small": ("sam2_hiera_s.yaml", [96, 192, 384, 768]),
        "large": ("sam2_hiera_l.yaml", [144, 288, 576, 1152]),
    }

    def __init__(
        self,
        class_num=1,
        mask_chans=1,
        checkpoint_path=None,
        freeze_backbone=True,
        image_size=352,
        backbone_variant="small",
        decoder_type="legacy",
        decoder_dim=96,
        use_detail_fusion=False,
        targetness_head=False,
        **kwargs,
    ):
        super().__init__()
        backbone_variant = backbone_variant.lower()
        if backbone_variant not in self.BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone_variant}; choose from {list(self.BACKBONES)}")
        model_cfg, feature_dims = self.BACKBONES[backbone_variant]
        if checkpoint_path is not None and not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint_path}")
        if checkpoint_path is None:
            warnings.warn(
                "SAM2 is randomly initialized. Set model.params.checkpoint_path before real training.",
                stacklevel=2,
            )
            freeze_backbone = False

        self.backbone = SAM2backbone(
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            image_size=image_size,
            model_cfg=model_cfg,
        )
        self.decoder_type = decoder_type
        if decoder_type == "high_resolution":
            self.decode_head = HighResolutionIRSTDDecoder(
                feature_dims=feature_dims,
                decoder_dim=decoder_dim,
                class_num=class_num,
                use_detail_fusion=use_detail_fusion,
                targetness_head=targetness_head,
            )
        elif decoder_type == "legacy":
            self.decode_head = Decoder(
                dims=feature_dims,
                dim=feature_dims[0],
                class_num=class_num,
                mask_chans=mask_chans,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    def forward(self, x, timesteps, cond_img):
        features = self.backbone(x, timesteps, cond_img)
        if self.decoder_type == "high_resolution":
            return self.decode_head(features, timesteps, x, cond_img)
        return self.decode_head(features, timesteps, x)

    def set_trainable_backbone_stages(
        self,
        last_n_stages,
        stage_indices=None,
        unfreeze_patch_embed=False,
    ):
        stage_ends = list(self.backbone.encoder.stage_ends)
        first_stage = max(0, len(stage_ends) - int(last_n_stages))
        first_block = 0 if first_stage == 0 else stage_ends[first_stage - 1] + 1
        selected_stages = set(int(index) for index in (stage_indices or []))
        for index, adapter in enumerate(self.backbone.encoder.blocks):
            stage_index = next(
                stage for stage, stage_end in enumerate(stage_ends) if index <= stage_end
            )
            train_block = index >= first_block or stage_index in selected_stages
            for parameter in adapter.block.parameters():
                parameter.requires_grad = train_block
            for parameter in adapter.prompt_learn.parameters():
                parameter.requires_grad = True
        if unfreeze_patch_embed:
            for parameter in self.backbone.encoder.patch_embed.parameters():
                parameter.requires_grad = True

    def optimizer_parameter_groups(self, head_lr, backbone_lr):
        backbone_parameters = []
        head_parameters = []
        for name, parameter in self.named_parameters():
            is_backbone_parameter = (
                name.startswith("backbone.encoder.")
                and ".prompt_learn." not in name
            )
            if is_backbone_parameter:
                backbone_parameters.append(parameter)
            else:
                head_parameters.append(parameter)
        return [
            {"params": head_parameters, "lr": head_lr, "name": "head"},
            {"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"},
        ]

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        return self.forward(x, timesteps, cond_img)

    def extract_features(self, cond_img):
        return cond_img


# Existing configuration paths and serialized model imports remain valid.
IRSTDNet = PDDiff
