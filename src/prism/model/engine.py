from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from ._kernel import SmoothedSoftmax, jsd, posterior
from ._typing import (
    DTYPE_NP,
    EPS,
    GeneBatch,
    GridDistribution,
    OPTIMIZERS,
    SCHEDULERS,
    OptimizerName,
    SchedulerName,
    TorchTensor,
)

__all__ = [
    "PriorEngine",
    "FitSummary",
    "PriorFitReport",
    "PriorEngineSetting",
    "PriorEngineTrainingConfig",
]

TorchDTypeName = Literal["float64", "float32"]
FitProgressCallback = Callable[[int, int, float, float, float, bool], None]


@dataclass(frozen=True, slots=True)
class FitSummary:
    gene_names: list[str]
    final_loss: float
    best_loss: float
    n_iter: int


@dataclass(frozen=True, slots=True)
class PriorFitReport:
    gene_names: list[str]
    support: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    init_q_hat: np.ndarray
    init_prior_weights: np.ndarray
    prior_weights: np.ndarray
    q_hat: np.ndarray
    final_loss: float
    best_loss: float
    loss_history: list[float]
    nll_history: list[float]
    align_history: list[float]
    setting: dict[str, object]
    training_config: dict[str, object]


@dataclass(frozen=True, slots=True)
class PriorEngineSetting:
    """PriorEngine 模型结构超参。"""

    grid_size: int = 512
    sigma_bins: float = 1.0
    align_loss_weight: float = 1.0
    torch_dtype: TorchDTypeName = "float64"


@dataclass(frozen=True, slots=True)
class PriorEngineTrainingConfig:
    """PriorEngine 优化过程配置。"""

    lr: float = 0.05
    n_iter: int = 100
    lr_min_ratio: float = 0.1
    grad_clip: float | None = None
    init_temperature: float = 1.0
    cell_chunk_size: int = 512
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"不支持的 optimizer: {self.optimizer!r}, 可选: {sorted(OPTIMIZERS)}"
            )
        if self.scheduler not in SCHEDULERS:
            raise ValueError(
                f"不支持的 scheduler: {self.scheduler!r}, 可选: {sorted(SCHEDULERS)}"
            )
        if self.cell_chunk_size < 1:
            raise ValueError(f"cell_chunk_size 必须 >= 1，收到 {self.cell_chunk_size}")


def _resolve_torch_dtype(dtype_name: TorchDTypeName) -> torch.dtype:
    match dtype_name:
        case "float64":
            return torch.float64
        case "float32":
            return torch.float32
        case _:
            raise ValueError(f"不支持的 torch dtype: {dtype_name!r}")


def _iter_cell_slices(n_cells: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_cells))
        for start in range(0, n_cells, chunk_size)
    ]


def _build_optimizer(
    params: list[torch.nn.Parameter],
    cfg: PriorEngineTrainingConfig,
) -> torch.optim.Optimizer:
    match cfg.optimizer:
        case "adam":
            return torch.optim.Adam(params, lr=cfg.lr)
        case "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr)
        case "sgd":
            return torch.optim.SGD(params, lr=cfg.lr)
        case "rmsprop":
            return torch.optim.RMSprop(params, lr=cfg.lr)
        case _:
            raise ValueError(f"不支持的 optimizer: {cfg.optimizer!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: PriorEngineTrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_iters = max(cfg.n_iter, 1)

    match cfg.scheduler:
        case "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_iters,
                eta_min=cfg.lr * cfg.lr_min_ratio,
            )
        case "linear":
            end_factor = max(cfg.lr_min_ratio, 0.0)
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=total_iters,
            )
        case "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=total_iters,
            )
        case "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(total_iters // 3, 1),
                gamma=max(cfg.lr_min_ratio, EPS),
            )
        case _:
            raise ValueError(f"不支持的 scheduler: {cfg.scheduler!r}")


class PriorEngine:
    def __init__(
        self,
        gene_names: list[str],
        setting: PriorEngineSetting = PriorEngineSetting(),
        device: torch.device | str = "cpu",
    ) -> None:
        if not gene_names:
            raise ValueError("gene_names 不能为空")
        if len(gene_names) != len(set(gene_names)):
            raise ValueError("gene_names 必须去重")
        if setting.grid_size < 2:
            raise ValueError(f"grid_size 必须 >= 2，收到 {setting.grid_size}")

        self.gene_names = list(gene_names)
        self.setting = setting
        self.device = torch.device(device)
        self.torch_dtype = _resolve_torch_dtype(setting.torch_dtype)

        self._gene_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
        self._logits = np.zeros(
            (len(self.gene_names), setting.grid_size), dtype=DTYPE_NP
        )
        self._grid_min = np.zeros(len(self.gene_names), dtype=DTYPE_NP)
        self._grid_max = np.ones(len(self.gene_names), dtype=DTYPE_NP)
        self._fitted = np.zeros(len(self.gene_names), dtype=bool)
        self._smoother = SmoothedSoftmax(
            sigma_bins=setting.sigma_bins,
            device=self.device,
        )
        self._base_grid_t = torch.linspace(
            0.0,
            1.0,
            setting.grid_size,
            dtype=self.torch_dtype,
            device=self.device,
        )

    @property
    def B(self) -> int:
        return len(self.gene_names)

    @property
    def fitted_genes(self) -> list[str]:
        return [name for name, idx in self._gene_to_idx.items() if self._fitted[idx]]

    def is_fitted(self, gene_name: str) -> bool:
        idx = self._require_gene(gene_name)
        return bool(self._fitted[idx])

    def is_all_fitted(self) -> bool:
        return bool(self._fitted.all())

    def fit(
        self,
        batch: GeneBatch,
        s_hat: float,
        training_cfg: PriorEngineTrainingConfig = PriorEngineTrainingConfig(),
        progress_callback: FitProgressCallback | None = None,
        grid_max_override: np.ndarray | None = None,
    ) -> FitSummary:
        report = self.fit_report(
            batch,
            s_hat,
            training_cfg,
            progress_callback=progress_callback,
            grid_max_override=grid_max_override,
        )
        return FitSummary(
            gene_names=report.gene_names,
            final_loss=report.final_loss,
            best_loss=report.best_loss,
            n_iter=training_cfg.n_iter,
        )

    def fit_report(
        self,
        batch: GeneBatch,
        s_hat: float,
        training_cfg: PriorEngineTrainingConfig = PriorEngineTrainingConfig(),
        progress_callback: FitProgressCallback | None = None,
        grid_max_override: np.ndarray | None = None,
    ) -> PriorFitReport:
        if s_hat <= 0:
            raise ValueError(f"s_hat 必须 > 0，收到 {s_hat}")
        batch.check_shape()
        self._validate_batch(batch, s_hat)

        indices = self._resolve_indices(batch.gene_names)
        if grid_max_override is None:
            grid_min, grid_max = self._compute_grid_bounds(batch, s_hat)
        else:
            grid_min = np.zeros(batch.B, dtype=DTYPE_NP)
            grid_max = np.asarray(grid_max_override, dtype=DTYPE_NP).reshape(-1)
            if grid_max.shape != (batch.B,):
                raise ValueError(
                    f"grid_max_override shape mismatch: expected {(batch.B,)}, got {grid_max.shape}"
                )
            if np.any(~np.isfinite(grid_max)) or np.any(grid_max <= 0):
                raise ValueError("grid_max_override must be finite and > 0")
            if np.any(grid_max > s_hat):
                raise ValueError("grid_max_override must satisfy grid_max <= s_hat")
        support = self._build_support_tensor(grid_min, grid_max)
        support_ratio = (support / s_hat).clamp(EPS, 1.0 - EPS)
        counts_t = torch.as_tensor(
            batch.counts.T,
            dtype=self.torch_dtype,
            device=self.device,
        )
        totals_t = torch.as_tensor(
            batch.totals,
            dtype=self.torch_dtype,
            device=self.device,
        )
        logits, init_q_hat, init_prior_weights = self._init_logits(
            indices, counts_t, totals_t, support_ratio, training_cfg
        )
        (
            best_logits,
            final_loss,
            best_loss,
            loss_history,
            nll_history,
            align_history,
            q_hat,
        ) = self._optimize(
            logits,
            counts_t,
            totals_t,
            support_ratio,
            training_cfg,
            progress_callback=progress_callback,
        )
        self._writeback(indices, best_logits, grid_min, grid_max)

        return PriorFitReport(
            gene_names=list(batch.gene_names),
            support=support.detach().cpu().numpy(),
            grid_min=grid_min.copy(),
            grid_max=grid_max.copy(),
            init_q_hat=init_q_hat,
            init_prior_weights=init_prior_weights,
            prior_weights=self._smoother(best_logits).detach().cpu().numpy(),
            q_hat=q_hat,
            final_loss=final_loss,
            best_loss=best_loss,
            loss_history=loss_history,
            nll_history=nll_history,
            align_history=align_history,
            setting=asdict(self.setting),
            training_config=asdict(training_cfg),
        )

    def get_priors(self, gene_names: str | list[str]) -> GridDistribution | None:
        single = isinstance(gene_names, str)
        names = [gene_names] if single else list(gene_names)
        indices = self._resolve_indices(names)
        if not np.all(self._fitted[indices]):
            return None

        logits_t = torch.as_tensor(
            self._logits[indices], dtype=self.torch_dtype, device=self.device
        )
        with torch.no_grad():
            weights = self._smoother(logits_t).detach().cpu().numpy()

        if single:
            return GridDistribution(
                grid_min=float(self._grid_min[indices[0]]),
                grid_max=float(self._grid_max[indices[0]]),
                weights=weights[0],
            )

        return GridDistribution(
            grid_min=self._grid_min[indices].copy(),
            grid_max=self._grid_max[indices].copy(),
            weights=weights,
        )

    def get_logits(self, gene_names: str | list[str]) -> np.ndarray | None:
        single = isinstance(gene_names, str)
        names = [gene_names] if single else list(gene_names)
        indices = self._resolve_indices(names)
        if not np.all(self._fitted[indices]):
            return None

        logits = self._logits[indices].copy()
        return logits[0] if single else logits

    def _require_gene(self, gene_name: str) -> int:
        if gene_name not in self._gene_to_idx:
            raise KeyError(f"基因 {gene_name!r} 不在 engine 中")
        return self._gene_to_idx[gene_name]

    def _resolve_indices(self, gene_names: list[str]) -> np.ndarray:
        return np.asarray(
            [self._require_gene(name) for name in gene_names], dtype=np.int64
        )

    def _validate_batch(self, batch: GeneBatch, s_hat: float) -> None:
        if batch.counts.ndim != 2:
            raise ValueError(f"counts 必须为二维，收到 shape={batch.counts.shape}")
        if batch.totals.ndim != 1:
            raise ValueError(f"totals 必须为一维，收到 shape={batch.totals.shape}")
        if len(batch.gene_names) != len(set(batch.gene_names)):
            raise ValueError("batch.gene_names 必须去重")
        if not np.all(np.isfinite(batch.counts)) or not np.all(
            np.isfinite(batch.totals)
        ):
            raise ValueError("counts 和 totals 必须全部为有限值")
        if np.any(batch.counts < 0) or np.any(batch.totals <= 0):
            raise ValueError("counts 不能为负，totals 必须为正")
        if np.any(batch.counts > batch.totals[:, None]):
            raise ValueError("要求逐元素满足 counts <= totals")
        if np.any(batch.counts > s_hat + 1e-12):
            raise ValueError("counts 不能超过 s_hat；请检查输入标度")

    def _compute_grid_bounds(
        self,
        batch: GeneBatch,
        s_hat: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        totals = np.maximum(batch.totals.astype(DTYPE_NP, copy=False), 1.0)
        counts = batch.counts.astype(DTYPE_NP, copy=False)
        empirical_max = np.max((counts / totals[:, None]) * s_hat, axis=0)

        grid_min = np.zeros(batch.B, dtype=DTYPE_NP)
        grid_max = np.clip(empirical_max, 1e-6, s_hat).astype(DTYPE_NP)
        return grid_min, grid_max

    def _build_support_tensor(
        self,
        grid_min: np.ndarray,
        grid_max: np.ndarray,
    ) -> TorchTensor:
        grid_min_t = torch.as_tensor(
            grid_min, dtype=self.torch_dtype, device=self.device
        )
        grid_max_t = torch.as_tensor(
            grid_max, dtype=self.torch_dtype, device=self.device
        )
        return (
            grid_min_t[:, None]
            + (grid_max_t - grid_min_t)[:, None] * self._base_grid_t[None, :]
        )

    def _log_lik_chunk(
        self,
        counts_chunk: TorchTensor,
        totals_chunk: TorchTensor,
        support_ratio: TorchTensor,
    ) -> TorchTensor:
        x = counts_chunk.unsqueeze(-1)
        n = totals_chunk.unsqueeze(0).unsqueeze(-1)
        p = support_ratio.unsqueeze(-2)
        log_coeff = (
            torch.lgamma(n + 1.0) - torch.lgamma(x + 1.0) - torch.lgamma(n - x + 1.0)
        )
        return log_coeff + x * torch.log(p) + (n - x) * torch.log1p(-p)

    def _init_logits(
        self,
        indices: np.ndarray,
        counts_t: TorchTensor,
        totals_t: TorchTensor,
        support_ratio: TorchTensor,
        cfg: PriorEngineTrainingConfig,
    ) -> tuple[torch.nn.Parameter, np.ndarray, np.ndarray]:
        init_logits = self._logits[indices].copy()
        cold_mask = ~self._fitted[indices]
        init_q_hat = np.full(
            (len(indices), self.setting.grid_size),
            1.0 / self.setting.grid_size,
            dtype=DTYPE_NP,
        )
        init_prior_weights = init_q_hat.copy()

        if np.any(cold_mask):
            cold_logits, cold_q_hat, cold_prior = (
                self._initialize_logits_from_posterior(
                    counts_t[cold_mask],
                    totals_t,
                    support_ratio[cold_mask],
                    cfg,
                )
            )
            init_logits[cold_mask] = cold_logits
            init_q_hat[cold_mask] = cold_q_hat
            init_prior_weights[cold_mask] = cold_prior

        if np.any(~cold_mask):
            fitted_logits = torch.as_tensor(
                init_logits[~cold_mask], dtype=self.torch_dtype, device=self.device
            )
            with torch.no_grad():
                fitted_prior = self._smoother(fitted_logits).detach().cpu().numpy()
            init_q_hat[~cold_mask] = fitted_prior
            init_prior_weights[~cold_mask] = fitted_prior

        logits_t = torch.as_tensor(
            init_logits, dtype=self.torch_dtype, device=self.device
        )
        return torch.nn.Parameter(logits_t.clone()), init_q_hat, init_prior_weights

    def _initialize_logits_from_posterior(
        self,
        counts_t: TorchTensor,
        totals_t: TorchTensor,
        support_ratio: TorchTensor,
        cfg: PriorEngineTrainingConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """未拟合基因显式初始化：一轮 posterior-generated distribution 反解析为 logits。"""
        with torch.no_grad():
            slices = _iter_cell_slices(totals_t.shape[0], cfg.cell_chunk_size)
            uniform_prior = torch.full(
                (counts_t.shape[0], support_ratio.shape[-1]),
                1.0 / support_ratio.shape[-1],
                dtype=counts_t.dtype,
                device=counts_t.device,
            )
            posterior_sum = torch.zeros_like(uniform_prior)
            for cell_slice in slices:
                log_lik = self._log_lik_chunk(
                    counts_t[:, cell_slice],
                    totals_t[cell_slice],
                    support_ratio,
                )
                if cfg.init_temperature != 1.0:
                    log_lik = log_lik / cfg.init_temperature
                post = posterior(log_lik, uniform_prior)
                posterior_sum += post.sum(dim=-2)
            init_weights = (posterior_sum / totals_t.shape[0]).clamp_min(EPS)
            init_weights = init_weights / init_weights.sum(dim=-1, keepdim=True)
            init_logits = torch.log(init_weights) / max(cfg.init_temperature, EPS)
            init_prior = self._smoother(init_logits)
            return (
                init_logits.cpu().numpy(),
                init_weights.cpu().numpy(),
                init_prior.cpu().numpy(),
            )

    def _optimize(
        self,
        logits: torch.nn.Parameter,
        counts_t: TorchTensor,
        totals_t: TorchTensor,
        support_ratio: TorchTensor,
        cfg: PriorEngineTrainingConfig,
        progress_callback: FitProgressCallback | None = None,
    ) -> tuple[
        TorchTensor,
        float,
        float,
        list[float],
        list[float],
        list[float],
        np.ndarray,
    ]:
        optimizer = _build_optimizer([logits], cfg)
        scheduler = _build_scheduler(optimizer, cfg)
        cell_slices = _iter_cell_slices(totals_t.shape[0], cfg.cell_chunk_size)

        best_loss = float("inf")
        best_logits = logits.detach().clone()
        final_loss = float("inf")
        loss_history: list[float] = []
        nll_history: list[float] = []
        align_history: list[float] = []
        best_q_hat: np.ndarray | None = None

        for _ in range(cfg.n_iter):
            optimizer.zero_grad(set_to_none=True)

            prior_weights = self._smoother(logits)
            log_prior = torch.log(prior_weights.clamp_min(EPS))

            with torch.no_grad():
                posterior_sum = torch.zeros_like(prior_weights)
                for cell_slice in cell_slices:
                    log_lik = self._log_lik_chunk(
                        counts_t[:, cell_slice],
                        totals_t[cell_slice],
                        support_ratio,
                    )
                    post = posterior(log_lik, prior_weights)
                    posterior_sum += post.sum(dim=-2)
                q_hat = (posterior_sum / totals_t.shape[0]).clamp_min(EPS)
                q_hat = q_hat / q_hat.sum(dim=-1, keepdim=True)

            align = jsd(q_hat, prior_weights).mean() * self.setting.align_loss_weight

            nll_value = 0.0
            total_loss = align
            for chunk_idx, cell_slice in enumerate(cell_slices):
                log_lik = self._log_lik_chunk(
                    counts_t[:, cell_slice],
                    totals_t[cell_slice],
                    support_ratio,
                )
                log_marginal = torch.logsumexp(
                    log_lik + log_prior.unsqueeze(-2), dim=-1
                )
                nll_chunk = -(log_marginal.sum(dim=-1) / totals_t.shape[0]).mean()
                total_loss = total_loss + nll_chunk
                nll_value += float(nll_chunk.item())

            total_loss.backward()

            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([logits], cfg.grad_clip)

            optimizer.step()
            scheduler.step()

            final_loss = nll_value + float(align.item())
            loss_history.append(final_loss)
            nll_history.append(nll_value)
            align_history.append(float(align.item()))
            if final_loss < best_loss:
                best_loss = final_loss
                best_logits = logits.detach().clone()
                best_q_hat = q_hat.detach().cpu().numpy()

            if progress_callback is not None:
                progress_callback(
                    len(loss_history),
                    cfg.n_iter,
                    final_loss,
                    nll_value,
                    float(align.item()),
                    final_loss <= best_loss,
                )

        if best_q_hat is None:
            best_q_hat = np.full(
                (logits.shape[0], logits.shape[1]),
                1.0 / logits.shape[1],
                dtype=DTYPE_NP,
            )

        return (
            best_logits,
            final_loss,
            best_loss,
            loss_history,
            nll_history,
            align_history,
            best_q_hat,
        )

    def _writeback(
        self,
        indices: np.ndarray,
        logits: TorchTensor,
        grid_min: np.ndarray,
        grid_max: np.ndarray,
    ) -> None:
        logits_np = logits.detach().cpu().numpy()
        self._logits[indices] = logits_np
        self._grid_min[indices] = grid_min
        self._grid_max[indices] = grid_max
        self._fitted[indices] = True
