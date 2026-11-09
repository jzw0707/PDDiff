import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class IRSTDDataset(Dataset):
    """SIRST4/SIRST-UAV reader using their img_idx split files and pixel masks."""

    def __init__(
        self,
        roots,
        split="train",
        image_size=352,
        training=None,
        preserve_aspect_ratio=True,
        horizontal_flip=0.5,
        vertical_flip=0.5,
        rotate_90=0.5,
        intensity_jitter=0.2,
        mean=None,
        std=None,
    ):
        self.roots = [Path(root) for root in ([roots] if isinstance(roots, str) else roots)]
        self.split = split
        self.image_size = image_size
        self.training = split == "train" if training is None else training
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_90 = rotate_90
        self.intensity_jitter = intensity_jitter
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.samples = []

        for root in self.roots:
            self.samples.extend(self._load_root(root))
        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} under {self.roots}")

    def _load_root(self, root):
        image_dir = root / "images"
        mask_dir = root / "masks"
        split_file = root / "img_idx" / f"{self.split}.txt"
        if not split_file.is_file():
            candidates = sorted((root / "img_idx").glob(f"{self.split}_*.txt"))
            if not candidates:
                raise FileNotFoundError(f"Cannot find a {self.split} split file under {root / 'img_idx'}")
            split_file = candidates[0]

        images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}
        sample_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        missing_images = [sample_id for sample_id in sample_ids if sample_id not in images]
        missing_masks = [sample_id for sample_id in sample_ids if sample_id not in masks]
        if missing_images or missing_masks:
            raise FileNotFoundError(
                f"Unmatched files in {root}: missing images={missing_images[:5]}, missing masks={missing_masks[:5]}"
            )
        return [(images[sample_id], masks[sample_id], root.name) for sample_id in sample_ids]

    def _augment(self, image, mask):
        if random.random() < self.horizontal_flip:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < self.vertical_flip:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        if random.random() < self.rotate_90:
            angle = random.choice((90, 180, 270))
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        if self.intensity_jitter > 0:
            low, high = 1 - self.intensity_jitter, 1 + self.intensity_jitter
            image = ImageEnhance.Brightness(image).enhance(random.uniform(low, high))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(low, high))
        return image, mask

    def _resize_and_pad(self, image, mask):
        width, height = image.size
        if not self.preserve_aspect_ratio:
            size = (self.image_size, self.image_size)
            return (
                TF.resize(image, size, interpolation=InterpolationMode.BILINEAR),
                TF.resize(mask, size, interpolation=InterpolationMode.NEAREST),
                (self.image_size, self.image_size),
                (0, 0),
            )

        scale = min(self.image_size / height, self.image_size / width)
        resized_h = max(1, min(self.image_size, round(height * scale)))
        resized_w = max(1, min(self.image_size, round(width * scale)))
        image = TF.resize(image, (resized_h, resized_w), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (resized_h, resized_w), interpolation=InterpolationMode.NEAREST)
        top = (self.image_size - resized_h) // 2
        left = (self.image_size - resized_w) // 2
        padding = (left, top, self.image_size - resized_w - left, self.image_size - resized_h - top)
        return TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=0), (resized_h, resized_w), (top, left)

    def __getitem__(self, index):
        image_path, mask_path, dataset_name = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("L")
        with Image.open(mask_path) as mask_file:
            original_mask = mask_file.convert("L")

        if image.size != original_mask.size:
            raise ValueError(f"Image/mask size mismatch: {image_path} and {mask_path}")
        original_size = (image.height, image.width)
        mask = original_mask.point(lambda value: 255 if value > 0 else 0)
        if self.training:
            image, mask = self._augment(image, mask)

        image, mask, resized_size, padding = self._resize_and_pad(image, mask)
        image = image.convert("RGB")
        image_tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        sample = {
            "image": image_tensor,
            "gt": mask_tensor if self.training else original_mask,
            "name": f"{image_path.stem}.png",
            "dataset": dataset_name,
        }
        if not self.training:
            sample.update(
                original_size=torch.tensor(original_size, dtype=torch.int64),
                resized_size=torch.tensor(resized_size, dtype=torch.int64),
                padding=torch.tensor(padding, dtype=torch.int64),
            )
        return sample

    def __len__(self):
        return len(self.samples)

