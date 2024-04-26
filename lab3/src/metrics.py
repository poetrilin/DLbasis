import torch
from torch import Tensor


def accuracy(preds: Tensor, y: Tensor) -> float:
    return (preds == y).sum().item() / len(y)
