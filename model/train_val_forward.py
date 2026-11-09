import torch
from torch import nn


def irstd_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        return model(gt, image, **kwargs)
    pred = model.sample(image, **kwargs)
    return {"image": image, "pred": pred, "gt": gt if gt is not None else None}


