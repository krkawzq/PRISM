"""scPRISM 的底层张量计算原语。"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from ._typing import DTYPE_TORCH, EPS, NEG_INF, TorchTensor

PaddingMode = Literal["replicate", "reflect", "zeros"]

__all__ = [
    "PaddingMode",
    "SmoothedSoftmax",
    "aggregate_posterior",
    "gaussian_kernel_1d",
    "jsd",
    "log_binomial_grid",
    "log_posterior",
    "map_histogram",
    "posterior",
]


def _ensure_2d(x: TorchTensor) -> tuple[TorchTensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    if x.ndim == 2:
        return x, False
    raise ValueError(f"要求输入为 (M,) 或 (B, M)，收到 shape={tuple(x.shape)}")


def gaussian_kernel_1d(sigma_bins: float) -> TorchTensor:
    """构建归一化的一维离散高斯核。"""
    if sigma_bins <= 0:
        return torch.ones(1, dtype=DTYPE_TORCH)

    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    offsets = torch.arange(-radius, radius + 1, dtype=DTYPE_TORCH)
    kernel = torch.exp(-0.5 * (offsets / sigma_bins) ** 2)
    return kernel / kernel.sum()


class SmoothedSoftmax:
    """`logits -> softmax -> gaussian smoothing -> renormalize`。"""

    def __init__(
        self,
        sigma_bins: float,
        padding: PaddingMode = "replicate",
        device: torch.device | None = None,
    ) -> None:
        self.padding = padding
        self.kernel = gaussian_kernel_1d(sigma_bins)
        if device is not None:
            self.kernel = self.kernel.to(device=device)
        self.radius = self.kernel.numel() // 2

    def _get_buffer(
        self,
        batch_size: int,
        grid_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TorchTensor:
        padded_size = grid_size + 2 * self.radius
        shape = (batch_size, 1, padded_size)
        return torch.empty(shape, dtype=dtype, device=device)

    def _fill_buffer(self, buffer: TorchTensor, weights: TorchTensor) -> None:
        radius = self.radius
        grid_size = weights.shape[-1]

        buffer[:, :, radius : radius + grid_size] = weights
        if radius == 0:
            return

        match self.padding:
            case "replicate":
                buffer[:, :, :radius] = weights[:, :, :1]
                buffer[:, :, radius + grid_size :] = weights[:, :, -1:]
            case "reflect":
                if grid_size < 2:
                    raise ValueError("reflect padding 要求网格长度至少为 2")

                left = weights[:, :, 1 : radius + 1].flip(dims=[-1])
                right = weights[:, :, -(radius + 1) : -1].flip(dims=[-1])

                if left.shape[-1] != radius or right.shape[-1] != radius:
                    raise ValueError(
                        "reflect padding 要求 grid_size > radius；请减小 sigma_bins 或改用 replicate"
                    )

                buffer[:, :, :radius] = left
                buffer[:, :, radius + grid_size :] = right
            case "zeros":
                buffer[:, :, :radius] = 0.0
                buffer[:, :, radius + grid_size :] = 0.0
            case _:
                raise ValueError(f"不支持的 padding: {self.padding!r}")

    def __call__(self, logits: TorchTensor) -> TorchTensor:
        logits_2d, unbatched = _ensure_2d(logits)
        weights = F.softmax(logits_2d, dim=-1)

        if self.kernel.numel() == 1:
            output = weights
            return output.squeeze(0).clone() if unbatched else output.clone()

        batch_size, grid_size = weights.shape
        kernel = self.kernel.to(device=weights.device, dtype=weights.dtype)
        buffer = self._get_buffer(
            batch_size,
            grid_size,
            dtype=weights.dtype,
            device=weights.device,
        )
        self._fill_buffer(buffer, weights.unsqueeze(1))

        smoothed = F.conv1d(buffer, kernel.view(1, 1, -1)).squeeze(1)
        smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return smoothed.squeeze(0).clone() if unbatched else smoothed.clone()


def log_binomial_grid(
    x: TorchTensor,
    n: TorchTensor,
    s_hat: float,
    support: TorchTensor,
) -> TorchTensor:
    """计算离散网格上的 Binomial 对数似然。"""
    if x.shape != n.shape:
        raise ValueError(
            f"x 和 n shape 必须一致，收到 {tuple(x.shape)} != {tuple(n.shape)}"
        )
    if support.ndim != 1:
        raise ValueError(f"support 必须为一维，收到 shape={tuple(support.shape)}")
    if s_hat <= 0:
        raise ValueError(f"s_hat 必须 > 0，收到 {s_hat}")

    x_col = x.unsqueeze(-1)
    n_col = n.unsqueeze(-1)
    support = support.to(device=x.device, dtype=x.dtype)
    p = (support / s_hat).clamp(EPS, 1.0 - EPS)

    log_coeff = (
        torch.lgamma(n_col + 1.0)
        - torch.lgamma(x_col + 1.0)
        - torch.lgamma(n_col - x_col + 1.0)
    )
    log_lik = log_coeff + x_col * torch.log(p) + (n_col - x_col) * torch.log1p(-p)
    invalid_mask = support > (s_hat + 1e-12)
    return log_lik.masked_fill(invalid_mask, NEG_INF)


def log_posterior(log_lik: TorchTensor, log_prior: TorchTensor) -> TorchTensor:
    """在 log 空间计算归一化后验。"""
    if log_lik.shape[-1] != log_prior.shape[-1]:
        raise ValueError(
            f"log_lik 和 log_prior 的最后一维必须一致，收到 {log_lik.shape[-1]} != {log_prior.shape[-1]}"
        )

    log_post = log_lik + log_prior.unsqueeze(-2)
    return log_post - torch.logsumexp(log_post, dim=-1, keepdim=True)


def posterior(log_lik: TorchTensor, prior_weights: TorchTensor) -> TorchTensor:
    """在概率空间计算后验。"""
    log_prior = torch.log(prior_weights.clamp_min(EPS))
    return torch.exp(log_posterior(log_lik, log_prior))


def aggregate_posterior(post: TorchTensor) -> TorchTensor:
    """将 per-cell 后验平均成群体分布。"""
    if post.ndim < 2:
        raise ValueError(
            f"post 至少应为二维 (..., C, M)，收到 shape={tuple(post.shape)}"
        )

    q_hat = post.mean(dim=-2)
    return q_hat / q_hat.sum(dim=-1, keepdim=True).clamp_min(EPS)


def map_histogram(post: TorchTensor) -> TorchTensor:
    """将 per-cell MAP 决策汇总为直方图分布。"""
    if post.ndim < 2:
        raise ValueError(
            f"post 至少应为二维 (..., C, M)，收到 shape={tuple(post.shape)}"
        )

    grid_size = post.shape[-1]
    map_idx = torch.argmax(post, dim=-1)

    if post.ndim == 2:
        q_hat = torch.zeros(grid_size, dtype=post.dtype, device=post.device)
        q_hat.scatter_add_(0, map_idx, torch.ones_like(map_idx, dtype=post.dtype))
        return q_hat / q_hat.sum().clamp_min(EPS)

    one_hot = F.one_hot(map_idx, num_classes=grid_size).to(dtype=post.dtype)
    q_hat = one_hot.mean(dim=-2)
    return q_hat / q_hat.sum(dim=-1, keepdim=True).clamp_min(EPS)


def jsd(p: TorchTensor, q: TorchTensor) -> TorchTensor:
    """计算 Jensen-Shannon 散度。"""
    if p.shape != q.shape:
        raise ValueError(
            f"p 和 q shape 必须一致，收到 {tuple(p.shape)} != {tuple(q.shape)}"
        )

    m = 0.5 * (p + q)
    kl_p = (p * torch.log((p + EPS) / (m + EPS))).sum(dim=-1)
    kl_q = (q * torch.log((q + EPS) / (m + EPS))).sum(dim=-1)
    return 0.5 * (kl_p + kl_q)
