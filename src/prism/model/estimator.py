from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import special

from ._typing import EPS, DTYPE_NP, PoolEstimate

__all__ = ["fit_pool_scale"]


@lru_cache(maxsize=16)
def _hermite_nodes_weights(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """缓存 Gauss-Hermite 求积节点与权重。"""
    if n_quad < 2:
        raise ValueError(f"n_quad 必须 >= 2，收到 {n_quad}")

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    return nodes.astype(DTYPE_NP), weights.astype(DTYPE_NP)


def _validate_totals(totals: np.ndarray) -> np.ndarray:
    totals = np.asarray(totals, dtype=DTYPE_NP).reshape(-1)
    if totals.size == 0:
        raise ValueError("totals 不能为空")
    if not np.all(np.isfinite(totals)):
        raise ValueError("totals 必须全部为有限值")
    if np.any(totals < 0):
        raise ValueError("totals 不能包含负数")
    return totals


def _compress_totals(totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将 totals 压缩为 (unique 值, 频率)。"""
    totals_int = np.rint(totals).astype(np.int64)
    unique_n, freq = np.unique(totals_int, return_counts=True)
    return unique_n.astype(DTYPE_NP), freq.astype(DTYPE_NP)


def _quadrature_log_joint(
    n_vals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算每个观测在求积节点上的 log joint。"""
    nodes, weights = _hermite_nodes_weights(n_quad)
    log_eta = mu + sigma * math.sqrt(2.0) * nodes
    eta = np.exp(log_eta)

    n_col = n_vals[:, None]
    log_poisson = n_col * log_eta[None, :] - eta[None, :] - special.gammaln(n_col + 1.0)
    log_weight = np.log(weights / math.sqrt(math.pi) + EPS)
    log_joint = log_poisson + log_weight[None, :]
    return log_eta, log_joint


def _stable_normalize_log_weights(
    log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对 log 权重做稳定归一化。"""
    log_max = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - log_max)
    normalizer = np.sum(weights, axis=1, keepdims=True) + EPS
    return weights / normalizer, log_max[:, 0] + np.log(normalizer[:, 0])


def _posterior_grid_weights(
    n_vals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 quadrature 网格上的后验权重。"""
    log_eta, log_joint = _quadrature_log_joint(n_vals, mu, sigma, n_quad)
    posterior_weight, _ = _stable_normalize_log_weights(log_joint)
    return log_eta, posterior_weight


def _log_marginal(
    n_vals: np.ndarray, mu: float, sigma: float, n_quad: int
) -> np.ndarray:
    """计算 log p(n | mu, sigma)。"""
    _, log_joint = _quadrature_log_joint(n_vals, mu, sigma, n_quad)
    _, log_marginal = _stable_normalize_log_weights(log_joint)
    return log_marginal


def _posterior_moments(
    n_vals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 E[log eta | n] 和 E[(log eta)^2 | n]。"""
    log_eta, posterior_weight = _posterior_grid_weights(n_vals, mu, sigma, n_quad)

    e_log_eta = np.sum(posterior_weight * log_eta[None, :], axis=1)
    e_log_eta2 = np.sum(posterior_weight * (log_eta[None, :] ** 2), axis=1)
    return e_log_eta, e_log_eta2


def _posterior_softargmax_mu(
    unique_n: np.ndarray,
    freq: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
    temperature: float,
) -> float:
    """从群体后验中提取 softargmax-MAP 风格的代表性 log-eta。"""
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，收到 {temperature}")

    log_eta, posterior_weight = _posterior_grid_weights(unique_n, mu, sigma, n_quad)
    population_posterior = np.sum(freq[:, None] * posterior_weight, axis=0)
    population_posterior = population_posterior / (np.sum(population_posterior) + EPS)

    logits = np.log(population_posterior + EPS) / temperature
    logits = logits - np.max(logits)
    soft_weight = np.exp(logits)
    soft_weight = soft_weight / (np.sum(soft_weight) + EPS)
    return float(np.sum(soft_weight * log_eta))


def _em_initialize(unique_n: np.ndarray, freq: np.ndarray) -> tuple[float, float]:
    """用压缩后的 totals 初始化 EM。"""
    log_n = np.log(np.maximum(unique_n, 1.0))
    mu_init = float(np.average(log_n, weights=freq))
    var_init = float(np.average((log_n - mu_init) ** 2, weights=freq))
    sigma_init = max(math.sqrt(var_init), 0.05)
    return mu_init, sigma_init


def _em_step(
    unique_n: np.ndarray,
    freq: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[float, float, float]:
    """执行一步 Poisson-LogNormal EM。"""
    e_log_eta, e_log_eta2 = _posterior_moments(unique_n, mu, sigma, n_quad)

    mu_new = float(np.average(e_log_eta, weights=freq))
    second_moment = float(np.average(e_log_eta2, weights=freq))
    sigma2_new = max(second_moment - mu_new**2, 1e-8)
    sigma_new = math.sqrt(sigma2_new)

    ll_unique = _log_marginal(unique_n, mu_new, sigma_new, n_quad)
    total_ll = float(np.sum(freq * ll_unique))
    return mu_new, sigma_new, total_ll


def fit_pool_scale(
    totals: np.ndarray,
    *,
    max_iter: int = 120,
    tol: float = 1e-6,
    n_quad: int = 128,
    use_posterior_mu: bool = False,
    softargmax_temperature: float = 0.05,
) -> PoolEstimate:
    """用 Poisson-LogNormal EM 拟合采样池标尺参数。"""
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，收到 {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol 必须 > 0，收到 {tol}")
    if use_posterior_mu and softargmax_temperature <= 0:
        raise ValueError(
            f"softargmax_temperature 必须 > 0，收到 {softargmax_temperature}"
        )

    totals = _validate_totals(totals)
    unique_n, freq = _compress_totals(totals)
    mu, sigma = _em_initialize(unique_n, freq)

    prev_ll = -np.inf
    for _ in range(max_iter):
        mu, sigma, ll = _em_step(unique_n, freq, mu, sigma, n_quad)
        if abs(ll - prev_ll) / (abs(prev_ll) + 1.0) < tol:
            break
        prev_ll = ll

    point_mu = mu
    if use_posterior_mu:
        point_mu = _posterior_softargmax_mu(
            unique_n,
            freq,
            mu,
            sigma,
            n_quad,
            softargmax_temperature,
        )

    return PoolEstimate(
        mu=mu,
        sigma=sigma,
        point_mu=point_mu,
        point_eta=float(np.exp(point_mu)),
        used_posterior_softargmax=use_posterior_mu,
    )
