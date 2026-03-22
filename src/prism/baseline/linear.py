from __future__ import annotations

import torch
from torch import nn

__all__ = ["LinearClassifier"]


class LinearClassifier(nn.Module):
    """最简单的线性分类 baseline。"""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim 必须 >= 1，收到 {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入最后一维必须等于 input_dim={self.input_dim}，收到 shape={tuple(x.shape)}"
            )
        return self.classifier(x)
