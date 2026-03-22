from __future__ import annotations

import torch
from torch import nn

__all__ = ["DeepMLPClassifier", "MLPClassifier"]


class MLPClassifier(nn.Module):
    """单隐藏层 MLP 分类 baseline。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim 必须 >= 1，收到 {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim 必须 >= 1，收到 {hidden_dim}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入最后一维必须等于 input_dim={self.input_dim}，收到 shape={tuple(x.shape)}"
            )
        return self.network(x)


class DeepMLPClassifier(nn.Module):
    """更深的多层 MLP 分类 baseline。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim 必须 >= 1，收到 {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}")
        if not hidden_dims:
            raise ValueError("hidden_dims 不能为空")
        if any(hidden_dim < 1 for hidden_dim in hidden_dims):
            raise ValueError(f"hidden_dims 必须全部 >= 1，收到 {hidden_dims}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims)

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入最后一维必须等于 input_dim={self.input_dim}，收到 shape={tuple(x.shape)}"
            )
        return self.network(x)
