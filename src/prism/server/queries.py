from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .router import Request
from .services.analysis import (
    AnalysisMode,
    GeneFitParams,
    KBulkParams,
    PriorSource,
)

Likelihood = Literal["binomial", "negative_binomial", "poisson"]


@dataclass(frozen=True, slots=True)
class HomePageQuery:
    search_query: str = ""
    browse_query: str = ""
    browse_scope: str = "auto"
    browse_sort: str = "total_count"
    browse_dir: str = "desc"
    browse_page: int = 1

    @classmethod
    def from_request(cls, request: Request) -> "HomePageQuery":
        return cls(
            search_query=(request.first("q") or "").strip(),
            browse_query=(request.first("browse_q") or "").strip(),
            browse_scope=(request.first("browse_scope") or "auto").strip().lower(),
            browse_sort=(request.first("browse_sort") or "total_count").strip().lower(),
            browse_dir=(request.first("browse_dir") or "desc").strip().lower(),
            browse_page=parse_int(request.first("browse_page"), default=1, min_value=1),
        )


@dataclass(frozen=True, slots=True)
class LoadRequestData:
    h5ad_path: str = ""
    ckpt_path: str = ""
    layer: str = ""

    @classmethod
    def from_request(cls, request: Request) -> "LoadRequestData":
        return cls(
            h5ad_path=(request.first("h5ad") or "").strip(),
            ckpt_path=(request.first("ckpt") or "").strip(),
            layer=(request.first("layer") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class GenePageQuery:
    query: str
    mode: AnalysisMode
    prior_source: PriorSource
    label_key: str | None
    label: str | None
    fit_params: GeneFitParams
    kbulk_params: KBulkParams
    run_kbulk: bool


def resolve_mode(request: Request) -> AnalysisMode:
    if (request.first("kbulk") or "") == "1":
        return "checkpoint"
    value = (request.first("mode") or "").strip().lower()
    if value in {"raw", "checkpoint", "fit"}:
        return cast(AnalysisMode, value)
    if (request.first("fit") or "") == "1":
        return "fit"
    return "checkpoint"


def resolve_prior_source(value: str | None) -> PriorSource:
    token = (value or "global").strip().lower()
    return cast(PriorSource, "label" if token == "label" else "global")


def parse_fit_params(request: Request) -> GeneFitParams:
    max_em_iterations_text = (request.first("max_em_iterations") or "").strip()
    return GeneFitParams(
        scale=parse_optional_float(request.first("scale")),
        reference_source="dataset"
        if (request.first("reference_source") or "").strip().lower() == "dataset"
        else "checkpoint",
        n_support_points=parse_int(
            request.first("n_support_points"), default=512, min_value=2
        ),
        max_em_iterations=(
            None
            if not max_em_iterations_text
            else parse_int(max_em_iterations_text, default=200, min_value=1)
        ),
        convergence_tolerance=parse_float(
            request.first("convergence_tolerance"),
            default=1e-6,
            min_value=0.0,
        ),
        cell_chunk_size=parse_int(
            request.first("cell_chunk_size"), default=512, min_value=1
        ),
        support_max_from="quantile"
        if (request.first("support_max_from") or "").strip().lower() == "quantile"
        else "observed_max",
        support_spacing="sqrt"
        if (request.first("support_spacing") or "").strip().lower() == "sqrt"
        else "linear",
        use_adaptive_support=parse_bool(
            request.first("use_adaptive_support"), default=False
        ),
        adaptive_support_fraction=parse_float(
            request.first("adaptive_support_fraction"),
            default=1.0,
            min_value=1e-12,
        ),
        adaptive_support_quantile_hi=parse_float(
            request.first("adaptive_support_quantile_hi"),
            default=0.99,
            min_value=1e-12,
        ),
        likelihood=cast(Likelihood, resolve_likelihood(request.first("likelihood"))),
        nb_overdispersion=parse_float(
            request.first("nb_overdispersion"),
            default=0.01,
            min_value=1e-12,
        ),
        torch_dtype="float32"
        if (request.first("torch_dtype") or "").strip().lower() == "float32"
        else "float64",
        compile_model=parse_bool(request.first("compile_model"), default=True),
        device=(request.first("device") or "cpu").strip() or "cpu",
    )


def parse_kbulk_params(
    request: Request,
    *,
    kbulk_default_max_classes: int,
) -> KBulkParams:
    return KBulkParams(
        class_key=(request.first("class_key") or "").strip() or None,
        k=parse_int(request.first("k"), default=8, min_value=1),
        n_samples=parse_int(request.first("n_samples"), default=24, min_value=1),
        sample_seed=parse_int(request.first("sample_seed"), default=0, min_value=0),
        max_classes=parse_int(
            request.first("max_classes"),
            default=kbulk_default_max_classes,
            min_value=1,
        ),
        sample_batch_size=parse_int(
            request.first("sample_batch_size"), default=32, min_value=1
        ),
        kbulk_prior_source=cast(
            PriorSource,
            resolve_prior_source(request.first("kbulk_prior_source")),
        ),
    )


def resolve_likelihood(value: str | None) -> Likelihood:
    token = (value or "binomial").strip().lower()
    if token in {"negative_binomial", "poisson"}:
        return cast(Likelihood, token)
    return "binomial"


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    token = value.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def parse_int(value: str | None, *, default: int, min_value: int | None = None) -> int:
    try:
        parsed = int((value or "").strip())
    except ValueError:
        parsed = default
    if min_value is not None and parsed < min_value:
        return min_value
    return parsed


def parse_float(
    value: str | None,
    *,
    default: float,
    min_value: float | None = None,
) -> float:
    try:
        parsed = float((value or "").strip())
    except ValueError:
        parsed = default
    if min_value is not None and parsed < min_value:
        return min_value
    return parsed


def parse_optional_float(value: str | None) -> float | None:
    token = (value or "").strip()
    if not token:
        return None
    return float(token)


__all__ = [
    "GenePageQuery",
    "HomePageQuery",
    "LoadRequestData",
    "parse_bool",
    "parse_fit_params",
    "parse_float",
    "parse_int",
    "parse_kbulk_params",
    "parse_optional_float",
    "resolve_likelihood",
    "resolve_mode",
    "resolve_prior_source",
]
